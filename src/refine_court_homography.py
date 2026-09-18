from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT
    / "Videos"
    / "test.mp4"
)

COARSE_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "auto_court.json"
)

LINES_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "court_detected_lines.json"
)

OUTPUT_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "refined_court.json"
)

OUTPUT_BIRDEYE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "refined_court_birdeye.jpg"
)

OUTPUT_ORIGINAL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "refined_court_original.jpg"
)


# ============================================================
# LOAD FILES
# ============================================================

if not COARSE_CALIBRATION_PATH.exists():

    raise RuntimeError(
        f"找不到：{COARSE_CALIBRATION_PATH}"
    )


if not LINES_PATH.exists():

    raise RuntimeError(
        f"找不到：{LINES_PATH}\n"
        "請先重新執行 detect_birdeye_lines.py"
    )


with open(
    COARSE_CALIBRATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    coarse_data = json.load(f)


with open(
    LINES_PATH,
    "r",
    encoding="utf-8"
) as f:

    line_data = json.load(f)


# ============================================================
# BASIC SETTINGS
# ============================================================

PIXELS_PER_METER = float(
    line_data["pixels_per_meter"]
)

PADDING = float(
    line_data["padding"]
)

OUTPUT_WIDTH = int(
    line_data["output_width"]
)

OUTPUT_HEIGHT = int(
    line_data["output_height"]
)


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40


# ============================================================
# COARSE HOMOGRAPHY
#
# Image pixel -> Court meter
# ============================================================

H_image_to_court_coarse = np.array(
    coarse_data["homography_image_to_court"],
    dtype=np.float64
)


# ============================================================
# Court meter -> Bird-eye pixel
# ============================================================

S = np.array(
    [
        [
            PIXELS_PER_METER,
            0,
            PADDING
        ],

        [
            0,
            PIXELS_PER_METER,
            PADDING
        ],

        [
            0,
            0,
            1
        ]
    ],
    dtype=np.float64
)


S_inv = np.linalg.inv(S)


# Coarse:
#
# image
#   ↓
# court meter
#   ↓
# bird-eye pixel

H_image_to_birdeye_coarse = (
    S
    @
    H_image_to_court_coarse
)


# ============================================================
# BUILD REFINEMENT POINT PAIRS
#
# CV detected intersection:
#
# (detected vertical x, detected horizontal y)
#
# should become:
#
# (ideal vertical x, ideal horizontal y)
#
# ============================================================

vertical_lines = (
    line_data["vertical_lines"]
)

horizontal_lines = (
    line_data["horizontal_lines"]
)


detected_points = []
ideal_points = []

point_names = []


for vertical_name, vertical_info in vertical_lines.items():

    if vertical_info["edge_hit"]:
        continue


    detected_x = float(
        vertical_info["detected_px"]
    )

    ideal_x = float(
        vertical_info["expected_px"]
    )


    for horizontal_name, horizontal_info in horizontal_lines.items():

        if horizontal_info["edge_hit"]:
            continue


        detected_y = float(
            horizontal_info["detected_px"]
        )

        ideal_y = float(
            horizontal_info["expected_px"]
        )


        detected_points.append(
            [
                detected_x,
                detected_y
            ]
        )


        ideal_points.append(
            [
                ideal_x,
                ideal_y
            ]
        )


        point_names.append(
            f"{vertical_name} x {horizontal_name}"
        )


detected_points = np.array(
    detected_points,
    dtype=np.float32
)

ideal_points = np.array(
    ideal_points,
    dtype=np.float32
)


print()
print("====================================")
print("CV REFINEMENT")
print("====================================")

print(
    "Intersection pairs：",
    len(detected_points)
)


if len(detected_points) < 4:

    raise RuntimeError(
        "有效 intersection 少於 4 個"
    )


# ============================================================
# REFINEMENT HOMOGRAPHY
#
# Current bird-eye
#        ↓
# Ideal bird-eye
# ============================================================

H_refine_birdeye, mask = cv2.findHomography(
    detected_points,
    ideal_points,

    method=cv2.RANSAC,

    ransacReprojThreshold=4.0
)


if H_refine_birdeye is None:

    raise RuntimeError(
        "Refinement Homography 計算失敗"
    )


mask = (
    mask
    .ravel()
    .astype(bool)
)


# Normalize
H_refine_birdeye = (
    H_refine_birdeye
    /
    H_refine_birdeye[2, 2]
)


# ============================================================
# REFINEMENT QUALITY
# ============================================================

projected_points = cv2.perspectiveTransform(
    detected_points.reshape(
        -1,
        1,
        2
    ),

    H_refine_birdeye
).reshape(
    -1,
    2
)


errors = np.linalg.norm(
    projected_points
    - ideal_points,

    axis=1
)


inlier_errors = errors[
    mask
]


print()
print(
    "========== REFINEMENT RANSAC =========="
)


for name, error, is_inlier in zip(
    point_names,
    errors,
    mask
):

    status = (
        "INLIER"
        if is_inlier
        else "OUTLIER"
    )

    print(
        f"{status:7s} "
        f"{error:6.2f}px "
        f"{name}"
    )


print()
print(
    f"Inliers: "
    f"{np.sum(mask)} / "
    f"{len(mask)}"
)


print(
    f"Median ALL error: "
    f"{np.median(errors):.3f}px"
)


if len(inlier_errors) > 0:

    print(
        f"Median INLIER error: "
        f"{np.median(inlier_errors):.3f}px"
    )

    print(
        f"Max INLIER error: "
        f"{np.max(inlier_errors):.3f}px"
    )


# ============================================================
# FINAL HOMOGRAPHY
#
# image
# ↓
# coarse court
# ↓ S
# coarse bird-eye
# ↓ H_refine
# refined bird-eye
# ↓ S^-1
# refined court meter
#
# Therefore:
#
# H_final =
#
# S^-1
# @ H_refine
# @ S
# @ H_coarse
# ============================================================

H_image_to_court_final = (
    S_inv
    @
    H_refine_birdeye
    @
    S
    @
    H_image_to_court_coarse
)


H_image_to_court_final = (
    H_image_to_court_final
    /
    H_image_to_court_final[2, 2]
)


H_court_to_image_final = np.linalg.inv(
    H_image_to_court_final
)


H_court_to_image_final = (
    H_court_to_image_final
    /
    H_court_to_image_final[2, 2]
)


# ============================================================
# SAVE FINAL CALIBRATION
# ============================================================

output_data = {

    "video":
        coarse_data.get(
            "video",
            VIDEO_PATH.name
        ),

    "method":
        "court_ai_multiframe_stretch_640"
        "+birdseye_cv_line_refinement",

    "pixels_per_meter":
        PIXELS_PER_METER,

    "padding":
        PADDING,

    "refinement_intersections":
        int(
            len(detected_points)
        ),

    "refinement_inliers":
        int(
            np.sum(mask)
        ),

    "refinement_median_error_px":
        float(
            np.median(errors)
        ),

    "refinement_median_inlier_error_px":
        float(
            np.median(
                inlier_errors
            )
        )
        if len(inlier_errors) > 0
        else None,

    "homography_image_to_court":
        H_image_to_court_final.tolist(),

    "homography_court_to_image":
        H_court_to_image_final.tolist(),

    "homography_refine_birdeye":
        H_refine_birdeye.tolist(),

    "source_coarse_calibration":
        str(
            COARSE_CALIBRATION_PATH.name
        ),

    "source_line_detection":
        str(
            LINES_PATH.name
        )
}


OUTPUT_CALIBRATION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_CALIBRATION_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# READ TEST FRAME
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        "影片讀取失敗"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 30


# 使用 2 秒畫面驗證
cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    int(
        fps * 2
    )
)


ok, frame = cap.read()

cap.release()


if not ok:

    raise RuntimeError(
        "測試 Frame 讀取失敗"
    )


# ============================================================
# FINAL IMAGE -> BIRD EYE
# ============================================================

H_image_to_birdeye_final = (
    S
    @
    H_image_to_court_final
)


refined_birdeye = cv2.warpPerspective(
    frame,

    H_image_to_birdeye_final,

    (
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT
    ),

    flags=cv2.INTER_LINEAR
)


# ============================================================
# DRAW IDEAL COURT
# ============================================================

def court_to_birdeye_pixel(
    x_m,
    y_m
):

    return (

        int(
            round(
                PADDING
                +
                x_m
                *
                PIXELS_PER_METER
            )
        ),

        int(
            round(
                PADDING
                +
                y_m
                *
                PIXELS_PER_METER
            )
        )
    )


def draw_birdeye_line(
    image,

    x1,
    y1,
    x2,
    y2,

    color=(0, 0, 255),

    thickness=2
):

    cv2.line(

        image,

        court_to_birdeye_pixel(
            x1,
            y1
        ),

        court_to_birdeye_pixel(
            x2,
            y2
        ),

        color,

        thickness,

        cv2.LINE_AA
    )


# outer
draw_birdeye_line(
    refined_birdeye,
    0,
    0,
    6.10,
    0
)

draw_birdeye_line(
    refined_birdeye,
    6.10,
    0,
    6.10,
    13.40
)

draw_birdeye_line(
    refined_birdeye,
    6.10,
    13.40,
    0,
    13.40
)

draw_birdeye_line(
    refined_birdeye,
    0,
    13.40,
    0,
    0
)


# singles sidelines
draw_birdeye_line(
    refined_birdeye,
    0.46,
    0,
    0.46,
    13.40
)

draw_birdeye_line(
    refined_birdeye,
    5.64,
    0,
    5.64,
    13.40
)


# doubles long service
draw_birdeye_line(
    refined_birdeye,
    0,
    0.76,
    6.10,
    0.76
)

draw_birdeye_line(
    refined_birdeye,
    0,
    12.64,
    6.10,
    12.64
)


# short service
draw_birdeye_line(
    refined_birdeye,
    0,
    4.72,
    6.10,
    4.72
)

draw_birdeye_line(
    refined_birdeye,
    0,
    8.68,
    6.10,
    8.68
)


# center
draw_birdeye_line(
    refined_birdeye,
    3.05,
    0,
    3.05,
    4.72
)

draw_birdeye_line(
    refined_birdeye,
    3.05,
    8.68,
    3.05,
    13.40
)


# ============================================================
# ORIGINAL FRAME VALIDATION
# ============================================================

original_overlay = frame.copy()


def court_to_original(
    x_m,
    y_m
):

    point = np.array(
        [
            [
                [
                    x_m,
                    y_m
                ]
            ]
        ],
        dtype=np.float32
    )


    projected = cv2.perspectiveTransform(
        point,
        H_court_to_image_final
    )


    return (

        int(
            round(
                projected[
                    0,
                    0,
                    0
                ]
            )
        ),

        int(
            round(
                projected[
                    0,
                    0,
                    1
                ]
            )
        )
    )


def draw_original_line(
    x1,
    y1,
    x2,
    y2,

    color=(0, 0, 255),

    thickness=2
):

    cv2.line(

        original_overlay,

        court_to_original(
            x1,
            y1
        ),

        court_to_original(
            x2,
            y2
        ),

        color,

        thickness,

        cv2.LINE_AA
    )


# outer
draw_original_line(
    0,
    0,
    6.10,
    0
)

draw_original_line(
    6.10,
    0,
    6.10,
    13.40
)

draw_original_line(
    6.10,
    13.40,
    0,
    13.40
)

draw_original_line(
    0,
    13.40,
    0,
    0
)


# singles
draw_original_line(
    0.46,
    0,
    0.46,
    13.40
)

draw_original_line(
    5.64,
    0,
    5.64,
    13.40
)


# doubles long service
draw_original_line(
    0,
    0.76,
    6.10,
    0.76
)

draw_original_line(
    0,
    12.64,
    6.10,
    12.64
)


# short service
draw_original_line(
    0,
    4.72,
    6.10,
    4.72
)

draw_original_line(
    0,
    8.68,
    6.10,
    8.68
)


# center service
draw_original_line(
    3.05,
    0,
    3.05,
    4.72
)

draw_original_line(
    3.05,
    8.68,
    3.05,
    13.40
)


# ============================================================
# SAVE IMAGES
# ============================================================

OUTPUT_BIRDEYE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


cv2.imwrite(
    str(
        OUTPUT_BIRDEYE_PATH
    ),
    refined_birdeye
)


cv2.imwrite(
    str(
        OUTPUT_ORIGINAL_PATH
    ),
    original_overlay
)


print()
print("====================================")
print("REFINEMENT COMPLETE")
print("====================================")

print()
print("Refined Calibration：")
print(
    OUTPUT_CALIBRATION_PATH
)

print()
print("Refined Bird-eye：")
print(
    OUTPUT_BIRDEYE_PATH
)

print()
print("Original Overlay：")
print(
    OUTPUT_ORIGINAL_PATH
)


# ============================================================
# SHOW BIRD EYE
# ============================================================

display_scale = min(

    900
    /
    OUTPUT_WIDTH,

    900
    /
    OUTPUT_HEIGHT,

    1.0
)


bird_display = cv2.resize(

    refined_birdeye,

    (
        int(
            OUTPUT_WIDTH
            *
            display_scale
        ),

        int(
            OUTPUT_HEIGHT
            *
            display_scale
        )
    )
)


cv2.imshow(
    "Refined Court - Bird Eye",
    bird_display
)


# ============================================================
# SHOW ORIGINAL
# ============================================================

frame_h, frame_w = (
    original_overlay.shape[:2]
)


original_scale = min(
    1400 / frame_w,
    850 / frame_h,
    1.0
)


original_display = cv2.resize(

    original_overlay,

    (
        int(
            frame_w
            *
            original_scale
        ),

        int(
            frame_h
            *
            original_scale
        )
    )
)


cv2.imshow(
    "Refined Court - Original",
    original_display
)


print()
print(
    "按任意鍵關閉"
)


cv2.waitKey(0)

cv2.destroyAllWindows()