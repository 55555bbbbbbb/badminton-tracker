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

# 優先使用上一階段 refinement
REFINED_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "refined_court.json"
)

COARSE_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "auto_court.json"
)

OUTPUT_CALIBRATION = (
    PROJECT_ROOT
    / "calibration"
    / "refined_court_v2.json"
)

OUTPUT_BACKGROUND = (
    PROJECT_ROOT
    / "outputs"
    / "linefit_background.jpg"
)

OUTPUT_DEBUG = (
    PROJECT_ROOT
    / "outputs"
    / "linefit_debug.jpg"
)

OUTPUT_BIRDEYE = (
    PROJECT_ROOT
    / "outputs"
    / "refined_v2_birdeye.jpg"
)

OUTPUT_ORIGINAL = (
    PROJECT_ROOT
    / "outputs"
    / "refined_v2_original.jpg"
)


# ============================================================
# SETTINGS
# ============================================================

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40

PIXELS_PER_METER = 120
PADDING = 40

# ------------------------------------------------------------
# 只使用同一段鏡頭附近建立背景
#
# Broadcast 影片可能中途切鏡頭，
# 所以不能從整支 405 秒平均抽 frame。
# ------------------------------------------------------------

CALIBRATION_START_SEC = 1.0
CALIBRATION_WINDOW_SEC = 8.0
NUM_BACKGROUND_FRAMES = 31

# 每個 sample 在理論線附近搜尋多少 pixel
SEARCH_RADIUS = 40

# 沿一條線每隔多少 pixel 取樣
SAMPLE_STEP = 30

# 每個 sample averaging band
SCAN_BAND_HALF = 4

# stripe center threshold
STRIPE_THRESHOLD_RATIO = 0.88

# 最少要有多少 sample 才能 fit 一條線
MIN_LINE_SAMPLES = 8

# 防止 peak 正好撞搜尋邊界
EDGE_MARGIN = 3


# ============================================================
# COURT LINE DEFINITIONS
# ============================================================

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


# ============================================================
# LOAD CURRENT CALIBRATION
# ============================================================

if REFINED_PATH.exists():

    CALIBRATION_PATH = REFINED_PATH

else:

    CALIBRATION_PATH = COARSE_PATH


if not CALIBRATION_PATH.exists():

    raise RuntimeError(
        "找不到 refined_court.json 或 auto_court.json"
    )


print()
print("使用目前 Calibration：")
print(CALIBRATION_PATH)


with open(
    CALIBRATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    calibration = json.load(f)


H_image_to_court_current = np.array(
    calibration["homography_image_to_court"],
    dtype=np.float64
)


# ============================================================
# COURT METER -> BIRD EYE PIXEL
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


H_image_to_birdeye_current = (
    S
    @
    H_image_to_court_current
)


OUTPUT_WIDTH = int(
    COURT_WIDTH_M
    * PIXELS_PER_METER
    + PADDING * 2
)


OUTPUT_HEIGHT = int(
    COURT_LENGTH_M
    * PIXELS_PER_METER
    + PADDING * 2
)


def meter_to_x(x_m):

    return float(
        PADDING
        + x_m * PIXELS_PER_METER
    )


def meter_to_y(y_m):

    return float(
        PADDING
        + y_m * PIXELS_PER_METER
    )


# ============================================================
# BUILD LOCAL MULTI-FRAME BACKGROUND
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"無法開啟影片：{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 30


frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


duration = (
    frame_count / fps
)


end_sec = min(
    CALIBRATION_START_SEC
    + CALIBRATION_WINDOW_SEC,

    duration - 0.1
)


sample_times = np.linspace(
    CALIBRATION_START_SEC,
    end_sec,
    NUM_BACKGROUND_FRAMES
)


warped_frames = []


print()
print("====================================")
print("建立 Local Bird-eye Background")
print("====================================")

print(
    f"Time range: "
    f"{CALIBRATION_START_SEC:.2f}s "
    f"→ {end_sec:.2f}s"
)


for second in sample_times:

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(second * fps)
    )


    ok, frame = cap.read()


    if not ok:
        continue


    warped = cv2.warpPerspective(
        frame,

        H_image_to_birdeye_current,

        (
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT
        ),

        flags=cv2.INTER_LINEAR
    )


    warped_frames.append(
        warped
    )


cap.release()


if len(warped_frames) < 10:

    raise RuntimeError(
        "建立背景成功取得的 frames 太少"
    )


print(
    "使用 Frames：",
    len(warped_frames)
)


stack = np.stack(
    warped_frames,
    axis=0
)


median_background = np.median(
    stack,
    axis=0
).astype(np.uint8)


OUTPUT_BACKGROUND.parent.mkdir(
    parents=True,
    exist_ok=True
)


cv2.imwrite(
    str(OUTPUT_BACKGROUND),
    median_background
)


# ============================================================
# WHITE SCORE
# ============================================================

hsv = cv2.cvtColor(
    median_background,
    cv2.COLOR_BGR2HSV
)


_, saturation, value = cv2.split(
    hsv
)


white_score = (
    value.astype(np.float32)
    -
    0.75
    *
    saturation.astype(np.float32)
)


white_score = np.clip(
    white_score,
    0,
    None
)


# ============================================================
# STRIPE CENTER DETECTOR
# ============================================================

def find_stripe_center(
    response,
    coordinate_start
):

    response = np.asarray(
        response,
        dtype=np.float32
    )


    if len(response) < 5:

        return None


    smooth = cv2.GaussianBlur(
        response.reshape(
            1,
            -1
        ),

        (
            5,
            1
        ),

        0
    ).reshape(-1)


    peak_index = int(
        np.argmax(smooth)
    )


    peak_value = float(
        smooth[peak_index]
    )


    edge_hit = (

        peak_index <= EDGE_MARGIN

        or

        peak_index
        >= len(smooth)
        - 1
        - EDGE_MARGIN
    )


    if edge_hit:

        return None


    # 白線 peak 是否真的比背景突出
    background_level = float(
        np.median(smooth)
    )


    prominence = (
        peak_value
        - background_level
    )


    if prominence < 5.0:

        return None


    threshold = (
        peak_value
        *
        STRIPE_THRESHOLD_RATIO
    )


    left = peak_index


    while (
        left > 0
        and
        smooth[left - 1]
        >= threshold
    ):

        left -= 1


    right = peak_index


    while (
        right
        < len(smooth) - 1

        and

        smooth[right + 1]
        >= threshold
    ):

        right += 1


    indices = np.arange(
        left,
        right + 1,
        dtype=np.float32
    )


    weights = smooth[
        left:right + 1
    ]


    weight_sum = float(
        np.sum(weights)
    )


    if weight_sum <= 0:

        center_local = float(
            peak_index
        )

    else:

        center_local = float(
            np.sum(
                indices * weights
            )
            /
            weight_sum
        )


    return (
        coordinate_start
        +
        center_local
    )


# ============================================================
# SAMPLE A VERTICAL LINE
# ============================================================

def sample_vertical_line(
    expected_x,
    y_ranges
):

    points = []


    for y_start, y_end in y_ranges:

        y = y_start


        while y <= y_end:

            yi = int(
                round(y)
            )


            y1 = max(
                0,
                yi - SCAN_BAND_HALF
            )

            y2 = min(
                OUTPUT_HEIGHT,
                yi + SCAN_BAND_HALF + 1
            )


            x1 = max(
                0,
                int(
                    round(
                        expected_x
                        -
                        SEARCH_RADIUS
                    )
                )
            )


            x2 = min(
                OUTPUT_WIDTH - 1,

                int(
                    round(
                        expected_x
                        +
                        SEARCH_RADIUS
                    )
                )
            )


            region = white_score[
                y1:y2,
                x1:x2 + 1
            ]


            if region.size > 0:

                response = np.mean(
                    region,
                    axis=0
                )


                detected_x = (
                    find_stripe_center(
                        response,
                        x1
                    )
                )


                if detected_x is not None:

                    points.append(
                        [
                            detected_x,
                            float(y)
                        ]
                    )


            y += SAMPLE_STEP


    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# SAMPLE A HORIZONTAL LINE
# ============================================================

def sample_horizontal_line(
    expected_y,
    x_start,
    x_end
):

    points = []

    x = x_start


    while x <= x_end:

        xi = int(
            round(x)
        )


        x1 = max(
            0,
            xi - SCAN_BAND_HALF
        )

        x2 = min(
            OUTPUT_WIDTH,
            xi + SCAN_BAND_HALF + 1
        )


        y1 = max(
            0,
            int(
                round(
                    expected_y
                    -
                    SEARCH_RADIUS
                )
            )
        )


        y2 = min(
            OUTPUT_HEIGHT - 1,

            int(
                round(
                    expected_y
                    +
                    SEARCH_RADIUS
                )
            )
        )


        region = white_score[
            y1:y2 + 1,
            x1:x2
        ]


        if region.size > 0:

            response = np.mean(
                region,
                axis=1
            )


            detected_y = (
                find_stripe_center(
                    response,
                    y1
                )
            )


            if detected_y is not None:

                points.append(
                    [
                        float(x),
                        detected_y
                    ]
                )


        x += SAMPLE_STEP


    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# LINE MATH
# ============================================================

def cv_fitline_to_abc(
    fit_result
):
    """
    OpenCV fitLine:
        vx, vy, x0, y0

    轉成：

        a*x + b*y + c = 0
    """

    vx, vy, x0, y0 = [
        float(v)
        for v in fit_result.flatten()
    ]


    a = vy
    b = -vx

    c = -(
        a * x0
        +
        b * y0
    )


    norm = np.sqrt(
        a * a
        +
        b * b
    )


    if norm <= 1e-9:

        raise RuntimeError(
            "無效 line fit"
        )


    return np.array(
        [
            a / norm,
            b / norm,
            c / norm
        ],
        dtype=np.float64
    )


def point_line_distances(
    points,
    line
):

    a, b, c = line


    return np.abs(
        a * points[:, 0]
        +
        b * points[:, 1]
        +
        c
    )


def robust_fit_line(
    points
):

    if len(points) < MIN_LINE_SAMPLES:

        return None


    working = points.copy()


    final_mask = np.ones(
        len(points),
        dtype=bool
    )


    # --------------------------------------------------------
    # 反覆 fit + MAD outlier removal
    # --------------------------------------------------------

    for _ in range(4):

        if len(working) < MIN_LINE_SAMPLES:

            return None


        fit = cv2.fitLine(
            working.reshape(
                -1,
                1,
                2
            ),

            cv2.DIST_HUBER,

            0,

            0.01,

            0.01
        )


        line = cv_fitline_to_abc(
            fit
        )


        distances = point_line_distances(
            points,
            line
        )


        median = float(
            np.median(distances)
        )


        mad = float(
            np.median(
                np.abs(
                    distances
                    -
                    median
                )
            )
        )


        robust_sigma = (
            1.4826
            *
            mad
        )


        threshold = max(
            2.0,

            median
            +
            3.0
            *
            robust_sigma
        )


        new_mask = (
            distances
            <= threshold
        )


        if (
            np.sum(new_mask)
            <
            MIN_LINE_SAMPLES
        ):

            break


        if np.array_equal(
            new_mask,
            final_mask
        ):

            final_mask = (
                new_mask
            )

            break


        final_mask = (
            new_mask
        )


        working = points[
            final_mask
        ]


    # --------------------------------------------------------
    # FINAL FIT
    # --------------------------------------------------------

    final_points = points[
        final_mask
    ]


    if (
        len(final_points)
        <
        MIN_LINE_SAMPLES
    ):

        return None


    fit = cv2.fitLine(
        final_points.reshape(
            -1,
            1,
            2
        ),

        cv2.DIST_HUBER,

        0,

        0.01,

        0.01
    )


    line = cv_fitline_to_abc(
        fit
    )


    residuals = point_line_distances(
        final_points,
        line
    )


    return {

        "line":
            line,

        "points":
            points,

        "inlier_mask":
            final_mask,

        "inlier_points":
            final_points,

        "median_residual":
            float(
                np.median(
                    residuals
                )
            ),

        "max_residual":
            float(
                np.max(
                    residuals
                )
            )
    }


def line_intersection(
    line1,
    line2
):

    p = np.cross(
        line1,
        line2
    )


    if abs(p[2]) < 1e-9:

        return None


    return np.array(
        [
            p[0] / p[2],
            p[1] / p[2]
        ],
        dtype=np.float32
    )


# ============================================================
# FIT ALL VERTICAL LINES
# ============================================================

vertical_fits = {}


print()
print("====================================")
print("VERTICAL LINE FITTING")
print("====================================")


for name, x_m in VERTICAL_LINES_M.items():

    expected_x = meter_to_x(
        x_m
    )


    if name == "center":

        y_ranges = [

            (
                meter_to_y(
                    0.10
                ),

                meter_to_y(
                    4.60
                )
            ),

            (
                meter_to_y(
                    8.80
                ),

                meter_to_y(
                    13.30
                )
            )
        ]

    else:

        y_ranges = [

            (
                meter_to_y(
                    0.10
                ),

                meter_to_y(
                    13.30
                )
            )
        ]


    points = sample_vertical_line(
        expected_x,
        y_ranges
    )


    fit = robust_fit_line(
        points
    )


    if fit is None:

        print(
            f"{name:20s} "
            f"FAILED "
            f"samples={len(points)}"
        )

        continue


    vertical_fits[
        name
    ] = fit


    print(
        f"{name:20s} "
        f"samples={len(points):3d} "
        f"inliers={len(fit['inlier_points']):3d} "
        f"median={fit['median_residual']:.2f}px "
        f"max={fit['max_residual']:.2f}px"
    )


# ============================================================
# FIT ALL HORIZONTAL LINES
# ============================================================

horizontal_fits = {}


print()
print("====================================")
print("HORIZONTAL LINE FITTING")
print("====================================")


x_start = meter_to_x(
    0.10
)

x_end = meter_to_x(
    6.00
)


for name, y_m in HORIZONTAL_LINES_M.items():

    expected_y = meter_to_y(
        y_m
    )


    points = sample_horizontal_line(
        expected_y,
        x_start,
        x_end
    )


    fit = robust_fit_line(
        points
    )


    if fit is None:

        print(
            f"{name:20s} "
            f"FAILED "
            f"samples={len(points)}"
        )

        continue


    horizontal_fits[
        name
    ] = fit


    print(
        f"{name:20s} "
        f"samples={len(points):3d} "
        f"inliers={len(fit['inlier_points']):3d} "
        f"median={fit['median_residual']:.2f}px "
        f"max={fit['max_residual']:.2f}px"
    )


# ============================================================
# CHECK WE HAVE ENOUGH LINES
# ============================================================

print()
print(
    "Vertical fitted:",
    len(vertical_fits),
    "/",
    len(VERTICAL_LINES_M)
)

print(
    "Horizontal fitted:",
    len(horizontal_fits),
    "/",
    len(HORIZONTAL_LINES_M)
)


if len(vertical_fits) < 4:

    raise RuntimeError(
        "成功 fitting 的 vertical lines 太少"
    )


if len(horizontal_fits) < 4:

    raise RuntimeError(
        "成功 fitting 的 horizontal lines 太少"
    )


# ============================================================
# REAL INTERSECTIONS
# ============================================================

detected_intersections = []
ideal_intersections = []
intersection_names = []


for vertical_name, vertical_fit in vertical_fits.items():

    ideal_x = meter_to_x(
        VERTICAL_LINES_M[
            vertical_name
        ]
    )


    for horizontal_name, horizontal_fit in horizontal_fits.items():

        ideal_y = meter_to_y(
            HORIZONTAL_LINES_M[
                horizontal_name
            ]
        )


        point = line_intersection(

            vertical_fit[
                "line"
            ],

            horizontal_fit[
                "line"
            ]
        )


        if point is None:

            continue


        # 避免超出合理區域太多
        if not (
            -50
            <= point[0]
            <= OUTPUT_WIDTH + 50

            and

            -50
            <= point[1]
            <= OUTPUT_HEIGHT + 50
        ):

            continue


        detected_intersections.append(
            point
        )


        ideal_intersections.append(
            [
                ideal_x,
                ideal_y
            ]
        )


        intersection_names.append(
            f"{vertical_name} "
            f"x "
            f"{horizontal_name}"
        )


detected_intersections = np.array(
    detected_intersections,
    dtype=np.float32
)


ideal_intersections = np.array(
    ideal_intersections,
    dtype=np.float32
)


print()
print("====================================")
print("REAL LINE INTERSECTIONS")
print("====================================")

print(
    "Intersections:",
    len(
        detected_intersections
    )
)


if len(detected_intersections) < 8:

    raise RuntimeError(
        "可用真實交點太少"
    )


# ============================================================
# REFINEMENT HOMOGRAPHY
# ============================================================

H_refine, mask = cv2.findHomography(

    detected_intersections,

    ideal_intersections,

    method=cv2.RANSAC,

    ransacReprojThreshold=3.5
)


if H_refine is None:

    raise RuntimeError(
        "Line-fit Homography refinement 失敗"
    )


H_refine = (
    H_refine
    /
    H_refine[2, 2]
)


mask = (
    mask
    .ravel()
    .astype(bool)
)


projected = cv2.perspectiveTransform(

    detected_intersections.reshape(
        -1,
        1,
        2
    ),

    H_refine

).reshape(
    -1,
    2
)


errors = np.linalg.norm(

    projected
    -
    ideal_intersections,

    axis=1
)


inlier_errors = errors[
    mask
]


print()
print("====================================")
print("LINE-FIT HOMOGRAPHY QUALITY")
print("====================================")


for name, error, good in zip(
    intersection_names,
    errors,
    mask
):

    status = (
        "INLIER "
        if good
        else "OUTLIER"
    )


    print(
        f"{status} "
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
    f"Median ALL: "
    f"{np.median(errors):.3f}px"
)


if len(inlier_errors) > 0:

    print(
        f"Median INLIER: "
        f"{np.median(inlier_errors):.3f}px"
    )

    print(
        f"Max INLIER: "
        f"{np.max(inlier_errors):.3f}px"
    )


# ============================================================
# FINAL HOMOGRAPHY
# ============================================================

H_image_to_court_final = (

    S_inv
    @
    H_refine
    @
    S
    @
    H_image_to_court_current
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
# SAVE CALIBRATION
# ============================================================

output_data = {

    "video":
        VIDEO_PATH.name,

    "method":
        "court_ai"
        "+stretch640"
        "+multiframe"
        "+birdseye_line_sampling"
        "+robust_line_fitting",

    "source_calibration":
        CALIBRATION_PATH.name,

    "calibration_start_sec":
        CALIBRATION_START_SEC,

    "calibration_end_sec":
        end_sec,

    "vertical_lines_fitted":
        list(
            vertical_fits.keys()
        ),

    "horizontal_lines_fitted":
        list(
            horizontal_fits.keys()
        ),

    "intersection_count":
        int(
            len(
                detected_intersections
            )
        ),

    "intersection_inliers":
        int(
            np.sum(mask)
        ),

    "median_all_error_px":
        float(
            np.median(errors)
        ),

    "median_inlier_error_px":
        float(
            np.median(
                inlier_errors
            )
        )
        if len(inlier_errors)
        else None,

    "max_inlier_error_px":
        float(
            np.max(
                inlier_errors
            )
        )
        if len(inlier_errors)
        else None,

    "homography_image_to_court":
        H_image_to_court_final.tolist(),

    "homography_court_to_image":
        H_court_to_image_final.tolist(),

    "homography_birdeye_refinement":
        H_refine.tolist()
}


OUTPUT_CALIBRATION.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_CALIBRATION,
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
# DEBUG IMAGE
# ============================================================

debug = (
    median_background.copy()
)


RED = (
    0,
    0,
    255
)

CYAN = (
    255,
    255,
    0
)

GREEN = (
    0,
    255,
    0
)

YELLOW = (
    0,
    255,
    255
)


# ============================================================
# DRAW IDEAL LINES
# ============================================================

court_top = int(
    round(
        meter_to_y(
            0
        )
    )
)

court_bottom = int(
    round(
        meter_to_y(
            13.40
        )
    )
)

court_left = int(
    round(
        meter_to_x(
            0
        )
    )
)

court_right = int(
    round(
        meter_to_x(
            6.10
        )
    )
)


for name, x_m in VERTICAL_LINES_M.items():

    x = int(
        round(
            meter_to_x(
                x_m
            )
        )
    )


    if name == "center":

        cv2.line(
            debug,
            (
                x,
                int(
                    meter_to_y(
                        0
                    )
                )
            ),
            (
                x,
                int(
                    meter_to_y(
                        4.72
                    )
                )
            ),
            RED,
            1
        )

        cv2.line(
            debug,
            (
                x,
                int(
                    meter_to_y(
                        8.68
                    )
                )
            ),
            (
                x,
                int(
                    meter_to_y(
                        13.40
                    )
                )
            ),
            RED,
            1
        )

    else:

        cv2.line(
            debug,
            (
                x,
                court_top
            ),
            (
                x,
                court_bottom
            ),
            RED,
            1
        )


for name, y_m in HORIZONTAL_LINES_M.items():

    y = int(
        round(
            meter_to_y(
                y_m
            )
        )
    )


    cv2.line(
        debug,
        (
            court_left,
            y
        ),
        (
            court_right,
            y
        ),
        RED,
        1
    )


# ============================================================
# DRAW SAMPLE POINTS + FITTED LINES
# ============================================================

def draw_vertical_fit(
    image,
    fit
):

    line = fit[
        "line"
    ]

    a, b, c = line


    y1 = float(
        court_top
    )

    y2 = float(
        court_bottom
    )


    if abs(a) < 1e-8:

        return


    x1 = -(
        b * y1
        +
        c
    ) / a


    x2 = -(
        b * y2
        +
        c
    ) / a


    cv2.line(
        image,

        (
            int(
                round(x1)
            ),
            int(
                round(y1)
            )
        ),

        (
            int(
                round(x2)
            ),
            int(
                round(y2)
            )
        ),

        CYAN,

        2,

        cv2.LINE_AA
    )


def draw_horizontal_fit(
    image,
    fit
):

    line = fit[
        "line"
    ]

    a, b, c = line


    x1 = float(
        court_left
    )

    x2 = float(
        court_right
    )


    if abs(b) < 1e-8:

        return


    y1 = -(
        a * x1
        +
        c
    ) / b


    y2 = -(
        a * x2
        +
        c
    ) / b


    cv2.line(
        image,

        (
            int(
                round(x1)
            ),
            int(
                round(y1)
            )
        ),

        (
            int(
                round(x2)
            ),
            int(
                round(y2)
            )
        ),

        CYAN,

        2,

        cv2.LINE_AA
    )


for name, fit in vertical_fits.items():

    points = fit[
        "points"
    ]


    mask_points = fit[
        "inlier_mask"
    ]


    for point, good in zip(
        points,
        mask_points
    ):

        color = (
            GREEN
            if good
            else YELLOW
        )


        cv2.circle(
            debug,
            (
                int(
                    round(
                        point[0]
                    )
                ),
                int(
                    round(
                        point[1]
                    )
                )
            ),
            2,
            color,
            -1
        )


    draw_vertical_fit(
        debug,
        fit
    )


for name, fit in horizontal_fits.items():

    points = fit[
        "points"
    ]


    mask_points = fit[
        "inlier_mask"
    ]


    for point, good in zip(
        points,
        mask_points
    ):

        color = (
            GREEN
            if good
            else YELLOW
        )


        cv2.circle(
            debug,
            (
                int(
                    round(
                        point[0]
                    )
                ),
                int(
                    round(
                        point[1]
                    )
                )
            ),
            2,
            color,
            -1
        )


    draw_horizontal_fit(
        debug,
        fit
    )


# 真實交點
for point, good in zip(
    detected_intersections,
    mask
):

    color = (
        GREEN
        if good
        else RED
    )


    cv2.circle(
        debug,
        (
            int(
                round(
                    point[0]
                )
            ),
            int(
                round(
                    point[1]
                )
            )
        ),
        5,
        color,
        -1
    )


cv2.imwrite(
    str(
        OUTPUT_DEBUG
    ),
    debug
)


# ============================================================
# FINAL BIRD-EYE VALIDATION
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    int(
        2.0
        *
        fps
    )
)


ok, test_frame = cap.read()

cap.release()


if not ok:

    raise RuntimeError(
        "驗證 frame 讀取失敗"
    )


H_image_to_birdeye_final = (

    S
    @
    H_image_to_court_final
)


final_birdeye = cv2.warpPerspective(

    test_frame,

    H_image_to_birdeye_final,

    (
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT
    )
)


# ============================================================
# DRAW IDEAL COURT ON FINAL BIRD-EYE
# ============================================================

def bird_pt(
    x_m,
    y_m
):

    return (
        int(
            round(
                meter_to_x(
                    x_m
                )
            )
        ),

        int(
            round(
                meter_to_y(
                    y_m
                )
            )
        )
    )


def bird_line(
    x1,
    y1,
    x2,
    y2
):

    cv2.line(
        final_birdeye,

        bird_pt(
            x1,
            y1
        ),

        bird_pt(
            x2,
            y2
        ),

        RED,

        2,

        cv2.LINE_AA
    )


# outer
bird_line(
    0,
    0,
    6.10,
    0
)

bird_line(
    6.10,
    0,
    6.10,
    13.40
)

bird_line(
    6.10,
    13.40,
    0,
    13.40
)

bird_line(
    0,
    13.40,
    0,
    0
)

# singles
bird_line(
    0.46,
    0,
    0.46,
    13.40
)

bird_line(
    5.64,
    0,
    5.64,
    13.40
)

# long service
bird_line(
    0,
    0.76,
    6.10,
    0.76
)

bird_line(
    0,
    12.64,
    6.10,
    12.64
)

# short service
bird_line(
    0,
    4.72,
    6.10,
    4.72
)

bird_line(
    0,
    8.68,
    6.10,
    8.68
)

# center
bird_line(
    3.05,
    0,
    3.05,
    4.72
)

bird_line(
    3.05,
    8.68,
    3.05,
    13.40
)


cv2.imwrite(
    str(
        OUTPUT_BIRDEYE
    ),
    final_birdeye
)


# ============================================================
# FINAL ORIGINAL OVERLAY
# ============================================================

H_court_to_image_final = (
    H_court_to_image_final
)


original_overlay = (
    test_frame.copy()
)


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


    result = cv2.perspectiveTransform(
        point,
        H_court_to_image_final
    )


    return (
        int(
            round(
                result[
                    0,
                    0,
                    0
                ]
            )
        ),

        int(
            round(
                result[
                    0,
                    0,
                    1
                ]
            )
        )
    )


def original_line(
    x1,
    y1,
    x2,
    y2
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

        RED,

        2,

        cv2.LINE_AA
    )


original_line(
    0,
    0,
    6.10,
    0
)

original_line(
    6.10,
    0,
    6.10,
    13.40
)

original_line(
    6.10,
    13.40,
    0,
    13.40
)

original_line(
    0,
    13.40,
    0,
    0
)

original_line(
    0.46,
    0,
    0.46,
    13.40
)

original_line(
    5.64,
    0,
    5.64,
    13.40
)

original_line(
    0,
    0.76,
    6.10,
    0.76
)

original_line(
    0,
    12.64,
    6.10,
    12.64
)

original_line(
    0,
    4.72,
    6.10,
    4.72
)

original_line(
    0,
    8.68,
    6.10,
    8.68
)

original_line(
    3.05,
    0,
    3.05,
    4.72
)

original_line(
    3.05,
    8.68,
    3.05,
    13.40
)


cv2.imwrite(
    str(
        OUTPUT_ORIGINAL
    ),
    original_overlay
)


# ============================================================
# RESULTS
# ============================================================

print()
print("====================================")
print("COURT V3C COMPLETE")
print("====================================")

print()
print("Final calibration：")
print(
    OUTPUT_CALIBRATION
)

print()
print("Line fitting debug：")
print(
    OUTPUT_DEBUG
)

print()
print("Final Bird-eye：")
print(
    OUTPUT_BIRDEYE
)

print()
print("Final Original Overlay：")
print(
    OUTPUT_ORIGINAL
)


# ============================================================
# SHOW
# ============================================================

debug_scale = min(
    900 / OUTPUT_WIDTH,
    900 / OUTPUT_HEIGHT,
    1.0
)


debug_display = cv2.resize(
    debug,

    (
        int(
            OUTPUT_WIDTH
            *
            debug_scale
        ),

        int(
            OUTPUT_HEIGHT
            *
            debug_scale
        )
    )
)


bird_display = cv2.resize(
    final_birdeye,

    (
        int(
            OUTPUT_WIDTH
            *
            debug_scale
        ),

        int(
            OUTPUT_HEIGHT
            *
            debug_scale
        )
    )
)


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
    "1 - Line Fitting Debug",
    debug_display
)

cv2.imshow(
    "2 - Refined V2 Bird Eye",
    bird_display
)

cv2.imshow(
    "3 - Refined V2 Original",
    original_display
)


print()
print(
    "按任意鍵關閉"
)


cv2.waitKey(0)

cv2.destroyAllWindows()