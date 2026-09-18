from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COURT_RESULTS_ROOT = PROJECT_ROOT / "results" / "court_benchmark"
REGRESSION_ROOT = PROJECT_ROOT / "results" / "singles_regression"

COURT_RUNNER = SRC_DIR / "run_court_test.py"
IDENTITY_RUNNER = SRC_DIR / "track_players_identity.py"
CAMERA_PROFILE_VALIDATOR = SRC_DIR / "camera_calibration_profile.py"


# ============================================================
# SINGLES ACCEPTANCE RULES
#
# Current logical-ID convention:
#   P1 / P2 = far side
#   P3 / P4 = near side
#
# CSV stores numeric logical IDs, so singles should only use 1 + 3 (displayed as P1 + P3).
# ============================================================

EXPECTED_SINGLES_PLAYERS = {"1", "3"}
UNEXPECTED_SINGLES_PLAYERS = {"2", "4"}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch regression for Videos/1p/1.mp4 ~ 9.mp4: "
            "court calibration + persistent identity + automatic singles checks."
        )
    )

    parser.add_argument(
        "--video-dir",
        default="Videos/1p",
        help="單打影片資料夾，預設 Videos/1p",
    )

    parser.add_argument(
        "--first",
        type=int,
        default=1,
        help="第一支 clip 編號，預設 1",
    )

    parser.add_argument(
        "--last",
        type=int,
        default=9,
        help="最後一支 clip 編號，預設 9",
    )

    parser.add_argument(
        "--force-court",
        action="store_true",
        help="即使已有且來源相符的 calibration，也重新跑 Court pipeline",
    )

    parser.add_argument(
        "--shared-calibration",
        default=None,
        help=(
            "所有 clips 共用同一份 refined_court_v2.json。"
            "適合同一固定攝影機 / 同一 framing 的多個 rally；"
            "指定後不會逐 clip 重跑 Court。"
        ),
    )

    parser.add_argument(
        "--camera-profile",
        default=None,
        help=(
            "使用 camera_calibration_profile.py 建立的 profile JSON。"
            "每支 clip 會先做快速 profile validation；PASS/WARN 才跑 Identity，"
            "FAIL 會跳過，避免盲目重用錯誤 calibration。"
        ),
    )

    parser.add_argument(
        "--reuse-identity",
        action="store_true",
        help="若已有 identity CSV 就重用；預設會用目前程式重新跑 identity",
    )

    parser.add_argument(
        "--debug-tracks",
        action="store_true",
        help="identity tracker 同時輸出 raw-track debug CSV",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def resolve_project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_command(command, log_path):
    """Run one subprocess, stream stdout, and save the same text to a log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("$", " ".join(str(part) for part in command))
    print()

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return process.wait()


def court_source_matches(summary, video_path):
    """Only reuse a court result when summary.json points to the same clip."""

    if not summary:
        return False

    recorded = summary.get("video")
    if not recorded:
        return False

    recorded_path = Path(recorded)

    return (
        recorded_path.name == video_path.name
        and recorded_path.parent.name == video_path.parent.name
    )


def classify_court_result(summary, calibration_path):
    """
    Batch-only court classification.

    We do NOT change run_court_test.py's historical verdict here.
    Known project behavior: a 5/6 horizontal fit can still have strong final
    geometry and a visually usable overlay, so this runner distinguishes a
    review warning from a hard failure.
    """

    if not calibration_path.exists():
        return "FAIL", "refined_court_v2.json missing"

    if not summary:
        return "FAIL", "summary.json missing/unreadable"

    original_status = summary.get("status")

    if original_status == "PIPELINE_FAIL":
        return "FAIL", summary.get("reason", "Court pipeline failed")

    if original_status == "GEOMETRY_PASS":
        return "PASS", "Court runner geometry passed"

    vertical = summary.get("vertical_lines")
    horizontal = summary.get("horizontal_lines")
    inliers = summary.get("inliers")
    median = summary.get("median_inlier_px")

    intersections = summary.get("intersections")
    max_inlier = summary.get("max_inlier_px")

    # Batch regression should distinguish a hard pipeline failure from a
    # completed Stage 4 whose historical verdict is merely too strict.
    # Current verified cases include 5/5 vertical + 4/6 horizontal with
    # 20 intersections / 15 inliers and sub-pixel median error. Those are
    # allowed to continue to Identity, but remain WARN_REVIEW until the
    # final overlay is visually checked.
    strong_completed_fit = (
        isinstance(vertical, int)
        and vertical >= 5
        and isinstance(horizontal, int)
        and horizontal >= 4
        and isinstance(intersections, int)
        and intersections >= 20
        and isinstance(inliers, int)
        and inliers >= 15
        and isinstance(median, (int, float))
        and median <= 3.0
        and (
            max_inlier is None
            or (isinstance(max_inlier, (int, float)) and max_inlier <= 5.0)
        )
    )

    if strong_completed_fit:
        return (
            "WARN_REVIEW",
            "Stage 4 completed and final metrics are numerically strong, but the historical geometry threshold failed; inspect refined_v2_original.jpg",
        )

    return "FAIL", summary.get("reason", "Court geometry failed")


def longest_consecutive_run(frames):
    values = sorted(set(frames))
    if not values:
        return 0

    longest = 1
    current = 1

    for previous, current_frame in zip(values, values[1:]):
        if current_frame == previous + 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1

    return max(longest, current)


def raw_id_transition_count(rows_for_player):
    """Raw ByteTrack ID changes are diagnostic; they are not identity failures."""

    ordered = sorted(rows_for_player, key=lambda row: int(row["frame"]))
    raw_ids = [
        row["raw_track_id"]
        for row in ordered
        if row.get("status") == "detected" and row.get("raw_track_id")
    ]

    if not raw_ids:
        return 0

    changes = 0
    previous = raw_ids[0]

    for raw_id in raw_ids[1:]:
        if raw_id != previous:
            changes += 1
            previous = raw_id

    return changes


def max_detection_gap(rows_for_player):
    detected_frames = sorted(
        {
            int(row["frame"])
            for row in rows_for_player
            if row.get("status") == "detected"
        }
    )

    if len(detected_frames) < 2:
        return 0

    return max(
        max(0, current - previous - 1)
        for previous, current in zip(detected_frames, detected_frames[1:])
    )


def raw_id_sort_key(value):
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def logical_id_label(value):
    text = str(value).strip()
    return f"P{text}" if text.isdigit() else text


def analyze_identity_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {
            "identity_status": "FAIL",
            "identity_reason": "identity CSV is empty",
        }

    rows_by_player = {}
    players_per_frame = {}
    detected_players_per_frame = {}
    predicted_frames_by_player = {}

    for row in rows:
        player_id = (row.get("player_id") or "").strip()
        if not player_id:
            continue

        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        rows_by_player.setdefault(player_id, []).append(row)
        players_per_frame.setdefault(frame, set()).add(player_id)

        if row.get("status") == "detected":
            detected_players_per_frame.setdefault(frame, set()).add(player_id)

        if row.get("status") == "predicted":
            predicted_frames_by_player.setdefault(player_id, []).append(frame)

    seen_players = set(rows_by_player)
    unexpected_players = sorted(seen_players - EXPECTED_SINGLES_PLAYERS)
    missing_expected = sorted(EXPECTED_SINGLES_PLAYERS - seen_players)

    max_logical_per_frame = max(
        (len(players) for players in players_per_frame.values()),
        default=0,
    )

    max_detected_logical_per_frame = max(
        (len(players) for players in detected_players_per_frame.values()),
        default=0,
    )

    longest_predicted = {
        player_id: longest_consecutive_run(frames)
        for player_id, frames in predicted_frames_by_player.items()
    }

    raw_ids_by_player = {}
    raw_id_transitions = {}
    detection_gap_by_player = {}

    for player_id, player_rows in sorted(rows_by_player.items()):
        raw_ids_by_player[player_id] = sorted(
            {
                row["raw_track_id"]
                for row in player_rows
                if row.get("status") == "detected" and row.get("raw_track_id")
            },
            key=raw_id_sort_key,
        )

        raw_id_transitions[player_id] = raw_id_transition_count(player_rows)
        detection_gap_by_player[player_id] = max_detection_gap(player_rows)

    fail_reasons = []

    if unexpected_players:
        fail_reasons.append(
            "unexpected logical players: " + ",".join(unexpected_players)
        )

    if max_logical_per_frame > 2:
        fail_reasons.append(
            f"max logical players in one frame = {max_logical_per_frame}"
        )

    if missing_expected:
        fail_reasons.append(
            "expected singles logical players never seen: "
            + ",".join(missing_expected)
        )

    identity_status = "FAIL" if fail_reasons else "PASS"
    identity_reason = (
        "; ".join(fail_reasons)
        if fail_reasons
        else "Singles logical roster stayed within P1/P3"
    )

    return {
        "identity_status": identity_status,
        "identity_reason": identity_reason,
        "players_seen": sorted(seen_players),
        "unexpected_players": unexpected_players,
        "missing_expected_players": missing_expected,
        "max_logical_per_frame": max_logical_per_frame,
        "max_detected_logical_per_frame": max_detected_logical_per_frame,
        "predicted_frame_count": sum(
            len(frames) for frames in predicted_frames_by_player.values()
        ),
        "longest_predicted_run_frames": longest_predicted,
        "max_detection_gap_frames": detection_gap_by_player,
        "raw_ids_by_player": raw_ids_by_player,
        "raw_id_transitions": raw_id_transitions,
    }


def write_summary_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "clip",
        "overall_status",
        "court_status",
        "court_original_status",
        "identity_status",
        "players_seen",
        "unexpected_players",
        "missing_expected_players",
        "max_logical_per_frame",
        "max_detected_logical_per_frame",
        "predicted_frame_count",
        "longest_predicted_run_frames",
        "max_detection_gap_frames",
        "raw_ids_by_player",
        "raw_id_transitions",
        "court_reason",
        "identity_reason",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in results:
            row = {}
            for field in fields:
                value = item.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                row[field] = value
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    video_dir = resolve_project_path(args.video_dir)
    shared_calibration = (
        resolve_project_path(args.shared_calibration)
        if args.shared_calibration
        else None
    )
    camera_profile = (
        resolve_project_path(args.camera_profile)
        if args.camera_profile
        else None
    )

    if args.first < 1 or args.last < args.first:
        raise ValueError("--first / --last 範圍不合法")

    if shared_calibration is not None and camera_profile is not None:
        raise ValueError(
            "--shared-calibration 與 --camera-profile 只能擇一。"
        )

    if (shared_calibration is not None or camera_profile is not None) and args.force_court:
        raise ValueError(
            "--shared-calibration / --camera-profile 與 --force-court 不能同時使用。"
        )

    if shared_calibration is not None and not shared_calibration.exists():
        raise FileNotFoundError(
            f"找不到 shared calibration：{shared_calibration}"
        )

    if camera_profile is not None and not camera_profile.exists():
        raise FileNotFoundError(
            f"找不到 camera profile：{camera_profile}"
        )

    if camera_profile is not None and not CAMERA_PROFILE_VALIDATOR.exists():
        raise FileNotFoundError(
            f"找不到 camera profile validator：{CAMERA_PROFILE_VALIDATOR}"
        )

    if not COURT_RUNNER.exists():
        raise FileNotFoundError(f"找不到 Court runner：{COURT_RUNNER}")

    if not IDENTITY_RUNNER.exists():
        raise FileNotFoundError(f"找不到 Identity runner：{IDENTITY_RUNNER}")

    # Only inspect the Court runner when this batch may actually execute Court.
    # Shared-calibration mode deliberately bypasses per-clip Court calibration.
    if shared_calibration is None and camera_profile is None:
        court_runner_text = COURT_RUNNER.read_text(encoding="utf-8")
        popen_count = court_runner_text.count("process = subprocess.Popen(")

        if popen_count != 1:
            raise RuntimeError(
                "run_court_test.py 可能仍有重複 subprocess.Popen。"
                f"目前整個檔案找到 {popen_count} 個 'process = subprocess.Popen('。"
                "請先使用 single-Popen 版本再跑 batch。"
            )

    REGRESSION_ROOT.mkdir(parents=True, exist_ok=True)

    batch_started = time.time()
    results = []

    for clip_number in range(args.first, args.last + 1):
        clip_name = f"{clip_number}.mp4"
        video_path = video_dir / clip_name
        run_name = f"1p_{clip_number:02d}"
        court_dir = COURT_RESULTS_ROOT / run_name
        calibration_path = court_dir / "refined_court_v2.json"
        court_summary_path = court_dir / "summary.json"

        clip_result_dir = REGRESSION_ROOT / run_name
        clip_result_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 72)
        print(f"SINGLES REGRESSION {clip_number}: {video_path}")
        print("=" * 72)

        result = {
            "clip": clip_name,
            "run_name": run_name,
            "video": str(video_path),
        }

        if not video_path.exists():
            result.update(
                {
                    "overall_status": "VIDEO_MISSING",
                    "court_status": "NOT_RUN",
                    "identity_status": "NOT_RUN",
                    "court_reason": "video missing",
                    "identity_reason": "video missing",
                }
            )
            results.append(result)
            write_json(clip_result_dir / "result.json", result)
            print("VIDEO_MISSING")
            continue

        # ----------------------------------------------------
        # COURT
        # ----------------------------------------------------

        if camera_profile is not None:
            calibration_path = camera_profile
            validation_json = clip_result_dir / "camera_profile_validation.json"
            validation_debug = clip_result_dir / "camera_profile_debug.jpg"
            validation_log = clip_result_dir / "camera_profile_command.log"

            validation_return_code = run_command(
                [
                    sys.executable,
                    CAMERA_PROFILE_VALIDATOR,
                    "validate",
                    "--profile",
                    camera_profile,
                    "--video",
                    video_path,
                    "--debug-image",
                    validation_debug,
                    "--json-output",
                    validation_json,
                ],
                validation_log,
            )

            profile_validation = load_json(validation_json)
            if not profile_validation:
                court_summary = None
                court_status = "FAIL"
                court_reason = (
                    "camera profile validator did not produce readable JSON "
                    f"(return code {validation_return_code})"
                )
            else:
                profile_status = profile_validation.get("status")
                if profile_status == "PASS":
                    court_status = "PROFILE_PASS"
                elif profile_status == "WARN_REVIEW":
                    court_status = "PROFILE_WARN"
                else:
                    court_status = "FAIL"
                court_reason = profile_validation.get("reason", "")
                court_summary = profile_validation

            result["camera_profile_validation"] = profile_validation
            result["camera_profile_validation_json"] = str(validation_json)
            result["camera_profile_debug"] = str(validation_debug)
            print(f"Camera profile: {camera_profile}")
            print(f"Profile validation: {court_status}")

        elif shared_calibration is not None:
            # Explicit operator decision: these clips share the same fixed
            # camera/framing, so Court calibration belongs to the camera
            # segment rather than to each individual rally.
            calibration_path = shared_calibration
            court_summary = None
            court_status = "SHARED"
            court_reason = (
                "Using explicitly supplied shared camera calibration; "
                "per-clip Court calibration skipped"
            )
            print(f"Shared court calibration: {calibration_path}")
        else:
            existing_summary = load_json(court_summary_path)
            can_reuse_court = (
                not args.force_court
                and calibration_path.exists()
                and court_source_matches(existing_summary, video_path)
            )

            if can_reuse_court:
                print(f"Reuse court calibration: {calibration_path}")
            else:
                court_log = clip_result_dir / "court_command.log"
                return_code = run_command(
                    [
                        sys.executable,
                        COURT_RUNNER,
                        video_path,
                        run_name,
                    ],
                    court_log,
                )

                if return_code != 0:
                    print(f"Court command return code: {return_code}")

            court_summary = load_json(court_summary_path)
            court_status, court_reason = classify_court_result(
                court_summary,
                calibration_path,
            )

        result["court_status"] = court_status
        result["court_reason"] = court_reason
        result["court_original_status"] = (
            court_summary.get("status") if court_summary else (
                "SHARED_CALIBRATION" if shared_calibration is not None else None
            )
        )
        result["court_summary"] = court_summary
        result["calibration"] = str(calibration_path)

        if court_status == "FAIL":
            result.update(
                {
                    "overall_status": "COURT_FAIL",
                    "identity_status": "NOT_RUN",
                    "identity_reason": "Identity skipped because Court calibration is not trustworthy",
                }
            )
            results.append(result)
            write_json(clip_result_dir / "result.json", result)
            print("COURT_FAIL -> skip identity")
            continue

        # ----------------------------------------------------
        # IDENTITY
        # ----------------------------------------------------

        video_group = video_path.parent.name
        identity_csv = (
            OUTPUTS_DIR
            / f"{video_group}_{video_path.stem}_identity_positions.csv"
        )
        identity_video = (
            OUTPUTS_DIR
            / f"{video_group}_{video_path.stem}_identity.mp4"
        )
        raw_tracks_csv = (
            OUTPUTS_DIR
            / f"{video_group}_{video_path.stem}_identity_raw_tracks.csv"
        )

        should_run_identity = not (
            args.reuse_identity and identity_csv.exists()
        )

        if should_run_identity:
            identity_command = [
                sys.executable,
                IDENTITY_RUNNER,
                "--video",
                video_path,
                "--calibration",
                calibration_path,
                "--match-format",
                "singles",
            ]

            if args.debug_tracks:
                identity_command.append("--debug-tracks")

            identity_log = clip_result_dir / "identity_command.log"
            identity_return_code = run_command(
                identity_command,
                identity_log,
            )

            if identity_return_code != 0:
                result.update(
                    {
                        "overall_status": "IDENTITY_ERROR",
                        "identity_status": "ERROR",
                        "identity_reason": (
                            "identity process return code = "
                            f"{identity_return_code}"
                        ),
                    }
                )
                results.append(result)
                write_json(clip_result_dir / "result.json", result)
                print("IDENTITY_ERROR")
                continue
        else:
            print(f"Reuse identity CSV: {identity_csv}")

        if not identity_csv.exists():
            result.update(
                {
                    "overall_status": "IDENTITY_ERROR",
                    "identity_status": "ERROR",
                    "identity_reason": (
                        "identity process finished but CSV is missing"
                    ),
                }
            )
            results.append(result)
            write_json(clip_result_dir / "result.json", result)
            continue

        identity_metrics = analyze_identity_csv(identity_csv)
        result.update(identity_metrics)
        result["identity_csv"] = str(identity_csv)
        result["identity_video"] = str(identity_video)
        result["identity_video_exists"] = identity_video.exists()

        if not identity_video.exists():
            result["overall_status"] = "IDENTITY_ERROR"
            result["identity_status"] = "ERROR"
            result["identity_reason"] = (
                "identity CSV exists but annotated MP4 is missing: "
                + str(identity_video)
            )
            results.append(result)
            write_json(clip_result_dir / "result.json", result)
            print("IDENTITY_ERROR -> annotated MP4 missing")
            continue

        if args.debug_tracks:
            result["raw_tracks_csv"] = str(raw_tracks_csv)

        if identity_metrics["identity_status"] == "FAIL":
            result["overall_status"] = "IDENTITY_FAIL"
        elif court_status in {"WARN_REVIEW", "PROFILE_WARN"}:
            result["overall_status"] = "PASS_IDENTITY_COURT_REVIEW"
        elif court_status == "PROFILE_PASS":
            result["overall_status"] = "PASS_CAMERA_PROFILE"
        elif court_status == "SHARED":
            result["overall_status"] = "PASS_SHARED_CALIBRATION"
        else:
            result["overall_status"] = "PASS"

        results.append(result)
        write_json(clip_result_dir / "result.json", result)

        print()
        print("Result:", result["overall_status"])
        print("Players seen:", identity_metrics.get("players_seen"))
        print("Unexpected:", identity_metrics.get("unexpected_players"))
        print(
            "Max logical/frame:",
            identity_metrics.get("max_logical_per_frame"),
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary_json_path = REGRESSION_ROOT / "summary.json"
    summary_csv_path = REGRESSION_ROOT / "summary.csv"

    summary_payload = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": time.time() - batch_started,
        "video_dir": str(video_dir),
        "clips": [args.first, args.last],
        "shared_calibration": (
            str(shared_calibration) if shared_calibration is not None else None
        ),
        "camera_profile": (
            str(camera_profile) if camera_profile is not None else None
        ),
        "acceptance": {
            "expected_logical_players": sorted(EXPECTED_SINGLES_PLAYERS),
            "unexpected_logical_players": sorted(
                UNEXPECTED_SINGLES_PLAYERS
            ),
            "max_logical_players_per_frame": 2,
            "raw_track_id_changes_are_allowed": True,
        },
        "results": results,
    }

    write_json(summary_json_path, summary_payload)
    write_summary_csv(summary_csv_path, results)

    print()
    print()
    print("=" * 72)
    print("SINGLES REGRESSION COMPLETE")
    print("=" * 72)
    print()

    for item in results:
        print(
            f"{item.get('clip', '?'):>8}  "
            f"{item.get('overall_status', '?'):<28}  "
            f"court={item.get('court_status', '?'):<11}  "
            f"players={','.join(logical_id_label(x) for x in item.get('players_seen', [])) or '-'}"
        )

    print()
    print("Annotated MP4s are written by the identity runner under:")
    print(OUTPUTS_DIR)
    if camera_profile is not None:
        print("Camera-profile mode: each clip was validated before Identity.")
        print("Camera profile:", camera_profile)
    elif shared_calibration is not None:
        print("Shared-calibration mode: per-clip Court calibration was skipped.")
        print("Shared calibration:", shared_calibration)
    else:
        print("COURT_FAIL clips intentionally skip identity, so they do not get an identity MP4.")
    print()
    print("Summary CSV:", summary_csv_path)
    print("Summary JSON:", summary_json_path)
    print()
    print(
        "注意：COURT WARN_REVIEW 仍需要人工看對應 "
        "refined_v2_original.jpg。"
    )
    print(
        "Identity automated PASS 不能單靠 CSV 證明所有視覺 continuity；"
        "異常/長 gap clip 應再看 MP4。"
    )


if __name__ == "__main__":
    main()
