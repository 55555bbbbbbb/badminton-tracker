from pathlib import Path
import argparse
import copy
import hashlib
import json
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
PIXELS_PER_METER = 120
PADDING = 40

CALIBRATION_START_SEC = 1.0
CALIBRATION_WINDOW_SEC = 8.0
NUM_BACKGROUND_FRAMES = 31

SEARCH_RADIUS = 40
SAMPLE_STEP = 30
SCAN_BAND_HALF = 4
STRIPE_THRESHOLD_RATIO = 0.88
MIN_LINE_SAMPLES = 8
EDGE_MARGIN = 3

# Camera-profile validation policy V1.
# These are not Court-fitting thresholds. They only judge whether an already
# verified homography still aligns with visible court lines in another clip.
# PASS allows up to 1/4 of the existing Stage-4 search radius as median drift;
# WARN allows up to 1/2. WARN is intentionally non-destructive and should be
# reviewed rather than treated as a new calibration automatically.
PASS_MAX_MEDIAN_OFFSET_PX = SEARCH_RADIUS * 0.25   # 10 px
WARN_MAX_MEDIAN_OFFSET_PX = SEARCH_RADIUS * 0.50   # 20 px
PASS_MIN_VERTICAL_LINES = 5
PASS_MIN_HORIZONTAL_LINES = 4
WARN_MIN_VERTICAL_LINES = 4
WARN_MIN_HORIZONTAL_LINES = 3

VERTICAL_LINES_M = {
    "left_doubles": 0.00,
    "left_singles": 0.46,
    "center": 3.05,
    "right_singles": 5.64,
    "right_doubles": 6.10,
}

HORIZONTAL_LINES_M = {
    "far_baseline": 0.00,
    "far_long_service": 0.76,
    "far_short_service": 4.72,
    "near_short_service": 8.68,
    "near_long_service": 12.64,
    "near_baseline": 13.40,
}


S = np.array(
    [
        [PIXELS_PER_METER, 0.0, PADDING],
        [0.0, PIXELS_PER_METER, PADDING],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

OUTPUT_WIDTH = int(COURT_WIDTH_M * PIXELS_PER_METER + PADDING * 2)
OUTPUT_HEIGHT = int(COURT_LENGTH_M * PIXELS_PER_METER + PADDING * 2)


def resolve_project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calibration_fingerprint(calibration):
    payload = {
        "homography_image_to_court": calibration["homography_image_to_court"],
        "homography_court_to_image": calibration["homography_court_to_image"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if frame_count > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
    }


def build_birdeye_background(video_path, h_image_to_court):
    meta = video_metadata(video_path)
    duration = meta["duration_sec"]

    if duration <= 0.2:
        raise RuntimeError(f"影片過短，無法驗證 camera profile：{video_path}")

    start_sec = min(CALIBRATION_START_SEC, max(0.0, duration * 0.10))
    end_sec = min(start_sec + CALIBRATION_WINDOW_SEC, duration - 0.1)
    if end_sec <= start_sec:
        start_sec = 0.0
        end_sec = max(0.1, duration - 0.05)

    sample_count = min(NUM_BACKGROUND_FRAMES, max(10, meta["frame_count"]))
    sample_times = np.linspace(start_sec, end_sec, sample_count)

    h_image_to_birdeye = S @ h_image_to_court
    cap = cv2.VideoCapture(str(video_path))
    warped_frames = []

    for second in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * meta["fps"]))
        ok, frame = cap.read()
        if not ok:
            continue

        warped = cv2.warpPerspective(
            frame,
            h_image_to_birdeye,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            flags=cv2.INTER_LINEAR,
        )
        warped_frames.append(warped)

    cap.release()

    if len(warped_frames) < 10:
        raise RuntimeError(
            f"建立 camera-profile background 的 frames 太少：{len(warped_frames)}"
        )

    background = np.median(np.stack(warped_frames, axis=0), axis=0).astype(np.uint8)
    return background, meta, {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "frames_used": len(warped_frames),
    }


def white_score_from_background(background):
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    white_score = value.astype(np.float32) - 0.75 * saturation.astype(np.float32)
    return np.clip(white_score, 0.0, None)


def find_stripe_center(response, coordinate_start):
    response = np.asarray(response, dtype=np.float32)
    if len(response) < 5:
        return None

    smooth = cv2.GaussianBlur(response.reshape(1, -1), (5, 1), 0).reshape(-1)
    peak_index = int(np.argmax(smooth))
    peak_value = float(smooth[peak_index])

    if (
        peak_index <= EDGE_MARGIN
        or peak_index >= len(smooth) - 1 - EDGE_MARGIN
        or peak_value <= 0
    ):
        return None

    threshold = peak_value * STRIPE_THRESHOLD_RATIO

    left = peak_index
    while left > 0 and smooth[left - 1] >= threshold:
        left -= 1

    right = peak_index
    while right < len(smooth) - 1 and smooth[right + 1] >= threshold:
        right += 1

    coordinates = np.arange(left, right + 1, dtype=np.float64)
    weights = smooth[left : right + 1].astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return None

    local_center = float(np.sum(coordinates * weights) / weight_sum)
    return float(coordinate_start + local_center)


def meter_to_x(x_m):
    return float(PADDING + x_m * PIXELS_PER_METER)


def meter_to_y(y_m):
    return float(PADDING + y_m * PIXELS_PER_METER)


def sample_vertical_line(white_score, expected_x):
    offsets = []

    for y in range(PADDING + 15, OUTPUT_HEIGHT - PADDING - 15, SAMPLE_STEP):
        x0 = int(round(expected_x))
        x_lo = max(0, x0 - SEARCH_RADIUS)
        x_hi = min(OUTPUT_WIDTH, x0 + SEARCH_RADIUS + 1)
        y_lo = max(0, y - SCAN_BAND_HALF)
        y_hi = min(OUTPUT_HEIGHT, y + SCAN_BAND_HALF + 1)

        region = white_score[y_lo:y_hi, x_lo:x_hi]
        if region.size == 0:
            continue

        response = np.mean(region, axis=0)
        center = find_stripe_center(response, x_lo)
        if center is not None:
            offsets.append(float(center - expected_x))

    return offsets


def sample_horizontal_line(white_score, expected_y):
    offsets = []

    for x in range(PADDING + 15, OUTPUT_WIDTH - PADDING - 15, SAMPLE_STEP):
        y0 = int(round(expected_y))
        y_lo = max(0, y0 - SEARCH_RADIUS)
        y_hi = min(OUTPUT_HEIGHT, y0 + SEARCH_RADIUS + 1)
        x_lo = max(0, x - SCAN_BAND_HALF)
        x_hi = min(OUTPUT_WIDTH, x + SCAN_BAND_HALF + 1)

        region = white_score[y_lo:y_hi, x_lo:x_hi]
        if region.size == 0:
            continue

        response = np.mean(region, axis=1)
        center = find_stripe_center(response, y_lo)
        if center is not None:
            offsets.append(float(center - expected_y))

    return offsets


def line_result(offsets):
    if not offsets:
        return {
            "samples": 0,
            "median_offset_px": None,
            "median_abs_offset_px": None,
            "max_abs_offset_px": None,
            "has_evidence": False,
        }

    values = np.asarray(offsets, dtype=np.float64)
    abs_values = np.abs(values)
    median_abs = float(np.median(abs_values))

    return {
        "samples": int(len(values)),
        "median_offset_px": float(np.median(values)),
        "median_abs_offset_px": median_abs,
        "max_abs_offset_px": float(np.max(abs_values)),
        "has_evidence": bool(
            len(values) >= MIN_LINE_SAMPLES
            and median_abs <= WARN_MAX_MEDIAN_OFFSET_PX
        ),
    }


def classify_metrics(vertical_lines, horizontal_lines):
    vertical_supported = [v for v in vertical_lines.values() if v["has_evidence"]]
    horizontal_supported = [v for v in horizontal_lines.values() if v["has_evidence"]]
    supported = vertical_supported + horizontal_supported

    line_medians = [
        item["median_abs_offset_px"]
        for item in supported
        if item["median_abs_offset_px"] is not None
    ]

    aggregate_median = float(np.median(line_medians)) if line_medians else None
    aggregate_max = float(max(line_medians)) if line_medians else None

    vertical_count = len(vertical_supported)
    horizontal_count = len(horizontal_supported)

    pass_geometry = (
        vertical_count >= PASS_MIN_VERTICAL_LINES
        and horizontal_count >= PASS_MIN_HORIZONTAL_LINES
        and aggregate_median is not None
        and aggregate_median <= PASS_MAX_MEDIAN_OFFSET_PX
    )

    warn_geometry = (
        vertical_count >= WARN_MIN_VERTICAL_LINES
        and horizontal_count >= WARN_MIN_HORIZONTAL_LINES
        and aggregate_median is not None
        and aggregate_median <= WARN_MAX_MEDIAN_OFFSET_PX
    )

    if pass_geometry:
        status = "PASS"
        reason = "shared calibration aligns with visible court-line evidence"
    elif warn_geometry:
        status = "WARN_REVIEW"
        reason = (
            "partial/looser court-line evidence; calibration may still be usable, "
            "but inspect debug overlay before trusting this camera segment"
        )
    else:
        status = "FAIL"
        reason = (
            "insufficient court-line evidence for this calibration; do not blindly "
            "reuse the camera profile on this clip"
        )

    return {
        "status": status,
        "reason": reason,
        "vertical_lines_with_evidence": vertical_count,
        "horizontal_lines_with_evidence": horizontal_count,
        "supported_line_count": len(supported),
        "median_line_offset_px": aggregate_median,
        "max_line_median_offset_px": aggregate_max,
    }


def draw_debug(background, vertical_lines, horizontal_lines):
    image = background.copy()

    for name, x_m in VERTICAL_LINES_M.items():
        x = int(round(meter_to_x(x_m)))
        info = vertical_lines[name]
        # Green = enough evidence, red = weak/missing. Debug only.
        color = (0, 220, 0) if info["has_evidence"] else (0, 0, 255)
        cv2.line(image, (x, PADDING), (x, OUTPUT_HEIGHT - PADDING), color, 2)

    for name, y_m in HORIZONTAL_LINES_M.items():
        y = int(round(meter_to_y(y_m)))
        info = horizontal_lines[name]
        color = (0, 220, 0) if info["has_evidence"] else (0, 0, 255)
        cv2.line(image, (PADDING, y), (OUTPUT_WIDTH - PADDING, y), color, 2)

    return image


def validate_calibration_on_video(calibration, video_path, expected_resolution=None, debug_image_path=None):
    h_image_to_court = np.array(
        calibration["homography_image_to_court"],
        dtype=np.float64,
    )

    background, meta, sampling = build_birdeye_background(video_path, h_image_to_court)
    white_score = white_score_from_background(background)

    vertical_lines = {
        name: line_result(sample_vertical_line(white_score, meter_to_x(x_m)))
        for name, x_m in VERTICAL_LINES_M.items()
    }
    horizontal_lines = {
        name: line_result(sample_horizontal_line(white_score, meter_to_y(y_m)))
        for name, y_m in HORIZONTAL_LINES_M.items()
    }

    classification = classify_metrics(vertical_lines, horizontal_lines)

    resolution_match = True
    if expected_resolution is not None:
        expected_w, expected_h = expected_resolution
        resolution_match = (
            int(meta["width"]) == int(expected_w)
            and int(meta["height"]) == int(expected_h)
        )
        if not resolution_match:
            classification["status"] = "FAIL"
            classification["reason"] = (
                "video resolution differs from the verified camera profile; "
                "homography cannot be reused blindly"
            )

    if debug_image_path is not None:
        debug_image_path = Path(debug_image_path)
        debug_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(debug_image_path),
            draw_debug(background, vertical_lines, horizontal_lines),
        )

    return {
        **classification,
        "video": str(video_path),
        "video_metadata": meta,
        "resolution_match": resolution_match,
        "sampling": sampling,
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
        "debug_image": str(debug_image_path) if debug_image_path is not None else None,
        "policy": {
            "search_radius_px": SEARCH_RADIUS,
            "min_line_samples": MIN_LINE_SAMPLES,
            "pass_max_median_offset_px": PASS_MAX_MEDIAN_OFFSET_PX,
            "warn_max_median_offset_px": WARN_MAX_MEDIAN_OFFSET_PX,
            "pass_min_vertical_lines": PASS_MIN_VERTICAL_LINES,
            "pass_min_horizontal_lines": PASS_MIN_HORIZONTAL_LINES,
            "warn_min_vertical_lines": WARN_MIN_VERTICAL_LINES,
            "warn_min_horizontal_lines": WARN_MIN_HORIZONTAL_LINES,
        },
    }


def create_profile(calibration_path, reference_video, output_path, name, verify_videos):
    calibration = load_json(calibration_path)

    reference_debug = output_path.with_name(output_path.stem + "_reference_debug.jpg")
    reference_validation = validate_calibration_on_video(
        calibration,
        reference_video,
        debug_image_path=reference_debug,
    )

    if reference_validation["status"] != "PASS":
        raise RuntimeError(
            "Reference video 沒有達到 PASS；拒絕把這份 calibration promotion 成 camera profile。"
        )

    ref_meta = reference_validation["video_metadata"]
    expected_resolution = (ref_meta["width"], ref_meta["height"])

    verification_results = []
    for index, video_path in enumerate(verify_videos, start=1):
        debug_path = output_path.with_name(
            output_path.stem + f"_verify_{index:02d}_debug.jpg"
        )
        result = validate_calibration_on_video(
            calibration,
            video_path,
            expected_resolution=expected_resolution,
            debug_image_path=debug_path,
        )
        verification_results.append(result)

    verification_failures = [
        result for result in verification_results if result["status"] == "FAIL"
    ]

    verification_warns = [
        result for result in verification_results if result["status"] == "WARN_REVIEW"
    ]

    if verification_failures:
        state = "REJECTED"
    elif verification_warns:
        state = "REVIEW_REQUIRED"
    elif verification_results:
        state = "VERIFIED"
    else:
        state = "CANDIDATE"

    profile = copy.deepcopy(calibration)
    profile["camera_profile"] = {
        "version": 1,
        "name": name,
        "state": state,
        "created_at_unix": time.time(),
        "calibration_fingerprint": calibration_fingerprint(calibration),
        "source_calibration": str(calibration_path),
        "reference_video": str(reference_video),
        "reference_resolution": [ref_meta["width"], ref_meta["height"]],
        "reference_validation": reference_validation,
        "verification_results": verification_results,
        "note": (
            "CANDIDATE = only reference clip checked. VERIFIED = one or more additional clips "
            "checked and all PASS. REVIEW_REQUIRED = at least one verify clip WARN. "
            "REJECTED = at least one verify clip FAIL. This JSON retains the normal "
            "calibration keys and can be passed directly to track_players_identity.py "
            "as --calibration."
        ),
    }

    write_json(output_path, profile)
    return profile


def validate_profile(profile_path, video_path, debug_image_path=None):
    profile = load_json(profile_path)
    profile_meta = profile.get("camera_profile")
    if not isinstance(profile_meta, dict):
        raise RuntimeError(
            "指定 JSON 沒有 camera_profile metadata；請先用 create 建立 profile。"
        )

    resolution = profile_meta.get("reference_resolution")
    expected_resolution = tuple(resolution) if resolution and len(resolution) == 2 else None

    return validate_calibration_on_video(
        profile,
        video_path,
        expected_resolution=expected_resolution,
        debug_image_path=debug_image_path,
    )


def print_result(result):
    print()
    print("Status:", result["status"])
    print("Reason:", result["reason"])
    print(
        "Lines:",
        f"vertical={result['vertical_lines_with_evidence']}/5",
        f"horizontal={result['horizontal_lines_with_evidence']}/6",
    )
    print("Median line offset:", result["median_line_offset_px"], "px")
    print("Max line median offset:", result["max_line_median_offset_px"], "px")
    print("Resolution match:", result["resolution_match"])
    if result.get("debug_image"):
        print("Debug image:", result["debug_image"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create/validate a reusable fixed-camera badminton court calibration profile."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="建立 camera calibration profile")
    create_parser.add_argument("--calibration", required=True)
    create_parser.add_argument("--reference-video", required=True)
    create_parser.add_argument(
        "--verify-video",
        action="append",
        default=[],
        help="額外用另一支同 camera clip 驗證；可重複指定",
    )
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--output", required=True)

    validate_parser = subparsers.add_parser("validate", help="驗證新 clip 是否可重用 profile")
    validate_parser.add_argument("--profile", required=True)
    validate_parser.add_argument("--video", required=True)
    validate_parser.add_argument("--debug-image", default=None)
    validate_parser.add_argument("--json-output", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "create":
        calibration_path = resolve_project_path(args.calibration)
        reference_video = resolve_project_path(args.reference_video)
        verify_videos = [resolve_project_path(v) for v in args.verify_video]
        output_path = resolve_project_path(args.output)

        for path in [calibration_path, reference_video, *verify_videos]:
            if not path.exists():
                raise FileNotFoundError(path)

        profile = create_profile(
            calibration_path,
            reference_video,
            output_path,
            args.name,
            verify_videos,
        )

        meta = profile["camera_profile"]
        print()
        print("Camera profile created:", output_path)
        print("State:", meta["state"])
        print("Fingerprint:", meta["calibration_fingerprint"])
        print_result(meta["reference_validation"])

        if meta["verification_results"]:
            print()
            print("Verification clips:")
            for item in meta["verification_results"]:
                print(
                    f"  {Path(item['video']).name}: {item['status']} "
                    f"V={item['vertical_lines_with_evidence']}/5 "
                    f"H={item['horizontal_lines_with_evidence']}/6 "
                    f"median={item['median_line_offset_px']}px"
                )

        if meta["state"] == "REJECTED":
            raise SystemExit(3)
        if meta["state"] == "REVIEW_REQUIRED":
            raise SystemExit(2)
        return

    profile_path = resolve_project_path(args.profile)
    video_path = resolve_project_path(args.video)
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    debug_image_path = (
        resolve_project_path(args.debug_image)
        if args.debug_image
        else PROJECT_ROOT / "outputs" / "camera_profile_validation" / f"{video_path.stem}_debug.jpg"
    )

    result = validate_profile(profile_path, video_path, debug_image_path)
    print_result(result)

    if args.json_output:
        write_json(resolve_project_path(args.json_output), result)

    if result["status"] == "FAIL":
        raise SystemExit(3)
    if result["status"] == "WARN_REVIEW":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
