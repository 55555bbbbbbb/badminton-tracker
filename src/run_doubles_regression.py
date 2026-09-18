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
REGRESSION_ROOT = PROJECT_ROOT / "results" / "doubles_regression"

IDENTITY_RUNNER = SRC_DIR / "track_players_identity.py"
CAMERA_PROFILE_VALIDATOR = SRC_DIR / "camera_calibration_profile.py"


# ============================================================
# DOUBLES ACCEPTANCE RULES
#
# Logical-ID convention:
#   P1 / P2 = far side
#   P3 / P4 = near side
#
# CSV stores numeric logical IDs, so doubles should contain 1,2,3,4.
# Automatic CSV checks cannot prove that P1/P2 or P3/P4 never visually swap;
# raw-ID transition frames and long prediction gaps are therefore surfaced as
# review points for the annotated MP4.
# ============================================================

EXPECTED_DOUBLES_PLAYERS = {"1", "2", "3", "4"}
MAX_LOGICAL_PLAYERS_PER_FRAME = 4
PREDICTION_REVIEW_THRESHOLD_FRAMES = 12


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch identity regression for Videos/2p/1.mp4 ~ 13.mp4 using "
            "one verified shared camera calibration."
        )
    )

    parser.add_argument(
        "--video-dir",
        default="Videos/2p",
        help="雙打影片資料夾，預設 Videos/2p",
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
        default=13,
        help="最後一支 clip 編號，預設 13",
    )

    calibration_group = parser.add_mutually_exclusive_group(required=True)

    calibration_group.add_argument(
        "--shared-calibration",
        default=None,
        help=(
            "同一固定攝影機共用的 refined_court_v2.json。"
            "不做 per-clip profile validation。"
        ),
    )

    calibration_group.add_argument(
        "--camera-profile",
        default=None,
        help=(
            "camera_calibration_profile.py 建立的 profile JSON。"
            "每支 clip 先 validate，再決定是否跑 Identity。"
        ),
    )

    parser.add_argument(
        "--reuse-identity",
        action="store_true",
        help="若已有 identity CSV 就重用；預設用目前程式重新跑 identity",
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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_command(command, log_path):
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


def consecutive_runs(frames):
    values = sorted(set(frames))
    if not values:
        return []

    runs = []
    start = values[0]
    previous = values[0]

    for frame in values[1:]:
        if frame == previous + 1:
            previous = frame
            continue

        runs.append((start, previous))
        start = frame
        previous = frame

    runs.append((start, previous))
    return runs


def raw_id_transition_details(rows_for_player):
    ordered = sorted(rows_for_player, key=lambda row: int(row["frame"]))
    detected = [
        row
        for row in ordered
        if row.get("status") == "detected" and row.get("raw_track_id")
    ]

    if not detected:
        return []

    transitions = []
    previous_raw_id = detected[0]["raw_track_id"]

    for row in detected[1:]:
        current_raw_id = row["raw_track_id"]
        if current_raw_id != previous_raw_id:
            transitions.append(
                {
                    "frame": int(row["frame"]),
                    "time_sec": float(row.get("time_sec") or 0.0),
                    "from_raw_id": previous_raw_id,
                    "to_raw_id": current_raw_id,
                }
            )
            previous_raw_id = current_raw_id

    return transitions


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
    unexpected_players = sorted(seen_players - EXPECTED_DOUBLES_PLAYERS)
    missing_expected = sorted(EXPECTED_DOUBLES_PLAYERS - seen_players)

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

    long_prediction_runs = {}
    for player_id, frames in predicted_frames_by_player.items():
        player_runs = []
        for start, end in consecutive_runs(frames):
            length = end - start + 1
            if length >= PREDICTION_REVIEW_THRESHOLD_FRAMES:
                player_runs.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                        "length_frames": length,
                    }
                )
        if player_runs:
            long_prediction_runs[player_id] = player_runs

    raw_ids_by_player = {}
    raw_id_transitions = {}
    raw_id_transition_frames = {}
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

        transition_details = raw_id_transition_details(player_rows)
        raw_id_transitions[player_id] = len(transition_details)
        raw_id_transition_frames[player_id] = transition_details
        detection_gap_by_player[player_id] = max_detection_gap(player_rows)

    fail_reasons = []

    if unexpected_players:
        fail_reasons.append(
            "unexpected logical players: " + ",".join(unexpected_players)
        )

    if max_logical_per_frame > MAX_LOGICAL_PLAYERS_PER_FRAME:
        fail_reasons.append(
            f"max logical players in one frame = {max_logical_per_frame}"
        )

    if missing_expected:
        fail_reasons.append(
            "expected doubles logical players never seen: "
            + ",".join(missing_expected)
        )

    identity_status = "FAIL" if fail_reasons else "PASS"
    identity_reason = (
        "; ".join(fail_reasons)
        if fail_reasons
        else "Doubles logical roster stayed within P1/P2/P3/P4"
    )

    review_flags = []

    transition_total = sum(raw_id_transitions.values())
    if transition_total > 0:
        review_flags.append(
            f"raw ID transitions present ({transition_total}); inspect transition frames for visual P# continuity"
        )

    if long_prediction_runs:
        review_flags.append(
            "long predicted runs present; inspect overlap/lost-detection recovery"
        )

    large_gap_players = {
        player_id: gap
        for player_id, gap in detection_gap_by_player.items()
        if gap >= PREDICTION_REVIEW_THRESHOLD_FRAMES
    }
    if large_gap_players:
        review_flags.append(
            "detection gaps >= 12 frames: "
            + ", ".join(
                f"P{player_id}={gap}"
                for player_id, gap in sorted(large_gap_players.items())
            )
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
        "long_prediction_runs": long_prediction_runs,
        "max_detection_gap_frames": detection_gap_by_player,
        "raw_ids_by_player": raw_ids_by_player,
        "raw_id_transitions": raw_id_transitions,
        "raw_id_transition_frames": raw_id_transition_frames,
        "manual_review_required": bool(review_flags),
        "review_flags": review_flags,
    }


def write_summary_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "clip",
        "overall_status",
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
        "manual_review_required",
        "review_flags",
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
    calibration_source = camera_profile or shared_calibration

    if args.first < 1 or args.last < args.first:
        raise ValueError("--first / --last 範圍不合法")

    if calibration_source is None or not calibration_source.exists():
        raise FileNotFoundError(
            f"找不到 calibration source：{calibration_source}"
        )

    if camera_profile is not None and not CAMERA_PROFILE_VALIDATOR.exists():
        raise FileNotFoundError(
            f"找不到 camera profile validator：{CAMERA_PROFILE_VALIDATOR}"
        )

    if not IDENTITY_RUNNER.exists():
        raise FileNotFoundError(f"找不到 Identity runner：{IDENTITY_RUNNER}")

    REGRESSION_ROOT.mkdir(parents=True, exist_ok=True)

    batch_started = time.time()
    results = []

    for clip_number in range(args.first, args.last + 1):
        clip_name = f"{clip_number}.mp4"
        video_path = video_dir / clip_name
        run_name = f"2p_{clip_number:02d}"

        clip_result_dir = REGRESSION_ROOT / run_name
        clip_result_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 72)
        print(f"DOUBLES REGRESSION {clip_number}: {video_path}")
        print("=" * 72)
        print(f"Calibration source: {calibration_source}")

        result = {
            "clip": clip_name,
            "run_name": run_name,
            "video": str(video_path),
            "court_status": "PROFILE_PENDING" if camera_profile is not None else "SHARED",
            "calibration": str(calibration_source),
        }

        if not video_path.exists():
            result.update(
                {
                    "overall_status": "VIDEO_MISSING",
                    "identity_status": "NOT_RUN",
                    "identity_reason": "video missing",
                }
            )
            results.append(result)
            write_json(clip_result_dir / "result.json", result)
            print("VIDEO_MISSING")
            continue

        if camera_profile is not None:
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

            profile_validation = None
            try:
                with open(validation_json, "r", encoding="utf-8") as f:
                    profile_validation = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

            result["camera_profile_validation"] = profile_validation
            result["camera_profile_validation_json"] = str(validation_json)
            result["camera_profile_debug"] = str(validation_debug)

            if not profile_validation:
                result.update(
                    {
                        "overall_status": "COURT_PROFILE_ERROR",
                        "court_status": "FAIL",
                        "identity_status": "NOT_RUN",
                        "identity_reason": "camera profile validation output missing/unreadable",
                    }
                )
                results.append(result)
                write_json(clip_result_dir / "result.json", result)
                continue

            profile_status = profile_validation.get("status")
            if profile_status == "FAIL":
                result.update(
                    {
                        "overall_status": "COURT_PROFILE_FAIL",
                        "court_status": "FAIL",
                        "identity_status": "NOT_RUN",
                        "identity_reason": "camera profile mismatch; Identity skipped",
                    }
                )
                results.append(result)
                write_json(clip_result_dir / "result.json", result)
                print("COURT_PROFILE_FAIL -> skip identity")
                continue
            elif profile_status == "WARN_REVIEW":
                result["court_status"] = "PROFILE_WARN"
            else:
                result["court_status"] = "PROFILE_PASS"

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
                calibration_source,
                "--match-format",
                "doubles",
            ]

            if args.debug_tracks:
                identity_command.append("--debug-tracks")

            identity_log = clip_result_dir / "identity_command.log"
            return_code = run_command(identity_command, identity_log)

            if return_code != 0:
                result.update(
                    {
                        "overall_status": "IDENTITY_ERROR",
                        "identity_status": "ERROR",
                        "identity_reason": (
                            "identity process return code = " f"{return_code}"
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
                    "identity_reason": "identity process finished but CSV is missing",
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
        elif result.get("court_status") == "PROFILE_WARN":
            result["overall_status"] = "PASS_AUTO_COURT_REVIEW"
        elif identity_metrics["manual_review_required"]:
            result["overall_status"] = "PASS_AUTO_REVIEW"
        elif result.get("court_status") == "PROFILE_PASS":
            result["overall_status"] = "PASS_CAMERA_PROFILE"
        else:
            result["overall_status"] = "PASS_AUTO"

        results.append(result)
        write_json(clip_result_dir / "result.json", result)

        print()
        print("Result:", result["overall_status"])
        print("Players seen:", identity_metrics.get("players_seen"))
        print("Review flags:", identity_metrics.get("review_flags"))

    summary_json_path = REGRESSION_ROOT / "summary.json"
    summary_csv_path = REGRESSION_ROOT / "summary.csv"

    summary_payload = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": time.time() - batch_started,
        "video_dir": str(video_dir),
        "clips": [args.first, args.last],
        "shared_calibration": (str(shared_calibration) if shared_calibration else None),
        "camera_profile": (str(camera_profile) if camera_profile else None),
        "acceptance": {
            "expected_logical_players": sorted(EXPECTED_DOUBLES_PLAYERS),
            "max_logical_players_per_frame": MAX_LOGICAL_PLAYERS_PER_FRAME,
            "raw_track_id_changes_are_allowed": True,
            "visual_swap_check_is_not_proven_by_csv": True,
        },
        "results": results,
    }

    write_json(summary_json_path, summary_payload)
    write_summary_csv(summary_csv_path, results)

    print()
    print()
    print("=" * 72)
    print("DOUBLES REGRESSION COMPLETE")
    print("=" * 72)
    print()

    for item in results:
        print(
            f"{item.get('clip', '?'):>8}  "
            f"{item.get('overall_status', '?'):<20}  "
            f"players={','.join(logical_id_label(x) for x in item.get('players_seen', [])) or '-':<14}  "
            f"review={'YES' if item.get('manual_review_required') else 'NO'}"
        )

    print()
    print("Annotated MP4s:", OUTPUTS_DIR)
    print("Summary CSV:", summary_csv_path)
    print("Summary JSON:", summary_json_path)
    print()
    print(
        "注意：CSV 可以自動抓 roster/lost/gap/raw-ID transition，"
        "但不能單獨證明同側 P1/P2 或 P3/P4 沒有視覺 swap。"
    )
    print(
        "PASS_AUTO_REVIEW 的 clips 請優先看 review_flags 對應的 transition/gap 時段。"
    )


if __name__ == "__main__":
    main()
