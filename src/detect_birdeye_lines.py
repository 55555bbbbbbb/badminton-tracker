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

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "auto_court.json"
)

OUTPUT_BACKGROUND = (
    PROJECT_ROOT
    / "outputs"
    / "court_median_background.jpg"
)

OUTPUT_DETECTION = (
    PROJECT_ROOT
    / "outputs"
    / "court_line_detection.jpg"
)

OUTPUT_LINES_JSON = (
    PROJECT_ROOT
    / "calibration"
    / "court_detected_lines.json"
)

# ============================================================
# COURT SETTINGS
# ============================================================

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40

PIXELS_PER_METER = 120
PADDING = 40

# 在理論線附近搜尋 ±45 px
SEARCH_RADIUS = 45

# 判定 peak 是否撞搜尋窗邊界
EDGE_MARGIN = 3

# 白線亮帶 threshold
# 越高 -> stripe 越窄
# 越低 -> stripe 越寬
STRIPE_THRESHOLD_RATIO = 0.90


# ============================================================
# LOAD HOMOGRAPHY
# ============================================================

if not CALIBRATION_PATH.exists():
    raise RuntimeError(
        f"找不到 calibration：{CALIBRATION_PATH}"
    )


with open(
    CALIBRATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    calibration = json.load(f)


H_image_to_court = np.array(
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


H_image_to_birdeye = (
    S @ H_image_to_court
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


# ============================================================
# HELPERS
# ============================================================

def meter_to_x(x_m):

    return int(
        round(
            PADDING
            + x_m * PIXELS_PER_METER
        )
    )


def meter_to_y(y_m):

    return int(
        round(
            PADDING
            + y_m * PIXELS_PER_METER
        )
    )


# ============================================================
# READ VIDEO
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


print()
print("====================================")
print("建立 Bird-eye Median Background")
print("====================================")

print(
    f"FPS：{fps:.2f}"
)

print(
    f"影片長度：{duration:.2f} 秒"
)


# ============================================================
# SAMPLE MULTIPLE FRAMES
# ============================================================

# 從整段影片平均取 21 個時間點
sample_times = np.linspace(
    1,
    max(
        1,
        duration - 1
    ),
    21
)


warped_frames = []


for second in sample_times:

    frame_number = int(
        second * fps
    )


    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )


    ok, frame = cap.read()


    if not ok:
        continue


    warped = cv2.warpPerspective(
        frame,

        H_image_to_birdeye,

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


if len(warped_frames) < 5:

    raise RuntimeError(
        "成功取得的 frame 太少"
    )


print()
print(
    "使用 Frame 數量：",
    len(warped_frames)
)


# ============================================================
# MEDIAN BACKGROUND
# ============================================================

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
# BUILD WHITE SCORE
# ============================================================

# 白線特性：
#
# Brightness 高
# Saturation 低
#
# 綠色地板通常：
#
# Saturation 高
#
# 所以：
#
# white_score = V - 0.75 * S


hsv = cv2.cvtColor(
    median_background,
    cv2.COLOR_BGR2HSV
)


_,S_channel,V_channel = cv2.split(
    hsv
)


V_float = V_channel.astype(
    np.float32
)


S_float = S_channel.astype(
    np.float32
)


white_score = (
    V_float
    - 0.75 * S_float
)


white_score = np.clip(
    white_score,
    0,
    None
)


# ============================================================
# COURT LINE DEFINITIONS
# ============================================================

VERTICAL_LINES = {

    "left_doubles":
        0.00,

    "left_singles":
        0.46,

    "center":
        3.05,

    "right_singles":
        5.64,

    "right_doubles":
        6.10,
}


HORIZONTAL_LINES = {

    "far_baseline":
        0.00,

    "far_long_service":
        0.76,

    "far_short_service":
        4.72,

    "near_short_service":
        8.68,

    "near_long_service":
        12.64,

    "near_baseline":
        13.40,
}


# ============================================================
# FIND STRIPE CENTER
# ============================================================

def find_stripe_center(
    response,
    coordinate_start
):
    """
    從一維 whiteness response 找白色 stripe 中心。

    不只找最亮 pixel，
    而是：

    1. smooth
    2. 找 peak
    3. 找 peak 周圍亮帶
    4. 用 whiteness weighted center 找 stripe 中心
    """

    response = np.asarray(
        response,
        dtype=np.float32
    )


    if len(response) < 5:

        return None


    # ========================================================
    # 1D SMOOTH
    #
    # 注意：
    # response 是橫向的一維 array
    #
    # reshape 成 1 x N
    #
    # kernel 用 (5, 1)
    # 才是真的沿 x 軸 smooth
    # ========================================================

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


    # ========================================================
    # FIND PEAK
    # ========================================================

    peak_index = int(
        np.argmax(
            smooth
        )
    )


    peak_value = float(
        smooth[
            peak_index
        ]
    )


    # ========================================================
    # EDGE HIT CHECK
    # ========================================================

    edge_hit = (

        peak_index
        <= EDGE_MARGIN

        or

        peak_index
        >= (
            len(smooth)
            - 1
            - EDGE_MARGIN
        )
    )


    # ========================================================
    # FIND WHITE STRIPE RANGE
    # ========================================================

    stripe_threshold = (

        peak_value
        * STRIPE_THRESHOLD_RATIO
    )


    left = peak_index


    while (
        left > 0
        and
        smooth[
            left - 1
        ]
        >= stripe_threshold
    ):

        left -= 1


    right = peak_index


    while (
        right
        < len(smooth) - 1

        and

        smooth[
            right + 1
        ]
        >= stripe_threshold
    ):

        right += 1


    # ========================================================
    # WEIGHTED STRIPE CENTER
    # ========================================================

    indices = np.arange(
        left,
        right + 1,
        dtype=np.float32
    )


    weights = smooth[
        left:right + 1
    ]


    weight_sum = float(
        np.sum(
            weights
        )
    )


    if weight_sum <= 0:

        center_local = float(
            peak_index
        )

    else:

        center_local = float(

            np.sum(
                indices
                * weights
            )

            /

            weight_sum
        )


    center_global = (

        coordinate_start
        + center_local
    )


    stripe_width = (

        right
        - left
        + 1
    )


    return {

        "center":
            float(
                center_global
            ),

        "peak":
            peak_value,

        "stripe_width":
            int(
                stripe_width
            ),

        "edge_hit":
            bool(
                edge_hit
            ),

        "local_peak":
            int(
                peak_index
            )
    }


# ============================================================
# DETECT VERTICAL
# ============================================================

def detect_vertical(
    expected_x,
    y1,
    y2
):

    x_start = max(
        0,
        expected_x
        - SEARCH_RADIUS
    )


    x_end = min(
        OUTPUT_WIDTH - 1,
        expected_x
        + SEARCH_RADIUS
    )


    region = white_score[
        y1:y2,
        x_start:x_end + 1
    ]


    if region.size == 0:

        return None


    # 每個 column 的平均白度
    response = np.mean(
        region,
        axis=0
    )


    return find_stripe_center(
        response,
        x_start
    )


# ============================================================
# DETECT HORIZONTAL
# ============================================================

def detect_horizontal(
    expected_y,
    x1,
    x2
):

    y_start = max(
        0,
        expected_y
        - SEARCH_RADIUS
    )


    y_end = min(
        OUTPUT_HEIGHT - 1,
        expected_y
        + SEARCH_RADIUS
    )


    region = white_score[
        y_start:y_end + 1,
        x1:x2
    ]


    if region.size == 0:

        return None


    # 每個 row 的平均白度
    response = np.mean(
        region,
        axis=1
    )


    return find_stripe_center(
        response,
        y_start
    )


# ============================================================
# COURT RANGE
# ============================================================

court_left = meter_to_x(
    0.00
)

court_right = meter_to_x(
    6.10
)

court_top = meter_to_y(
    0.00
)

court_bottom = meter_to_y(
    13.40
)


# Horizontal line 搜尋時，
# 不用最左右邊緣，
# 避免場外雜訊

horizontal_x1 = (
    court_left
    + 25
)


horizontal_x2 = (
    court_right
    - 25
)


detected_verticals = {}

detected_horizontals = {}


# ============================================================
# DETECT VERTICAL LINES
# ============================================================

print()
print(
    "========== VERTICAL LINES =========="
)


for name, x_m in VERTICAL_LINES.items():

    expected_x = meter_to_x(
        x_m
    )


    # ========================================================
    # CENTER SERVICE LINE
    #
    # 中央線不是整場都有
    #
    # 所以：
    # 遠半場測一次
    # 近半場測一次
    #
    # 最後取 median
    # ========================================================

    if name == "center":

        y_ranges = [

            (
                meter_to_y(
                    0.40
                ),

                meter_to_y(
                    4.50
                )
            ),

            (
                meter_to_y(
                    8.90
                ),

                meter_to_y(
                    13.00
                )
            )
        ]


        sub_results = []


        for y1, y2 in y_ranges:

            r = detect_vertical(
                expected_x,
                y1,
                y2
            )


            if r is not None:

                sub_results.append(
                    r
                )


        valid_results = [

            r
            for r in sub_results

            if not r[
                "edge_hit"
            ]
        ]


        if not valid_results:

            print(
                f"{name:20s} "
                f"DETECTION FAILED / EDGE HIT"
            )

            continue


        detected_x = float(
            np.median(
                [
                    r["center"]
                    for r
                    in valid_results
                ]
            )
        )


        strength = float(
            np.mean(
                [
                    r["peak"]
                    for r
                    in valid_results
                ]
            )
        )


        stripe_width = int(
            round(
                np.mean(
                    [
                        r[
                            "stripe_width"
                        ]

                        for r
                        in valid_results
                    ]
                )
            )
        )


        edge_hit = False


    else:

        result = detect_vertical(
            expected_x,

            court_top + 20,

            court_bottom - 20
        )


        if result is None:

            print(
                f"{name:20s} "
                f"DETECTION FAILED"
            )

            continue


        detected_x = float(
            result[
                "center"
            ]
        )


        strength = float(
            result[
                "peak"
            ]
        )


        stripe_width = int(
            result[
                "stripe_width"
            ]
        )


        edge_hit = bool(
            result[
                "edge_hit"
            ]
        )


    detected_verticals[
        name
    ] = {

        "position":
            detected_x,

        "edge_hit":
            edge_hit
    }


    offset = (
        detected_x
        - expected_x
    )


    status = (

        "EDGE_HIT"

        if edge_hit

        else "OK"
    )


    print(

        f"{name:20s} "

        f"expected="
        f"{expected_x:7.2f} "

        f"detected="
        f"{detected_x:7.2f} "

        f"offset="
        f"{offset:+7.2f}px "

        f"width="
        f"{stripe_width:2d}px "

        f"score="
        f"{strength:6.1f} "

        f"{status}"
    )


# ============================================================
# DETECT HORIZONTAL LINES
# ============================================================

print()
print(
    "========== HORIZONTAL LINES =========="
)


for name, y_m in HORIZONTAL_LINES.items():

    expected_y = meter_to_y(
        y_m
    )


    result = detect_horizontal(

        expected_y,

        horizontal_x1,

        horizontal_x2
    )


    if result is None:

        print(
            f"{name:20s} "
            f"DETECTION FAILED"
        )

        continue


    detected_y = float(
        result[
            "center"
        ]
    )


    strength = float(
        result[
            "peak"
        ]
    )


    stripe_width = int(
        result[
            "stripe_width"
        ]
    )


    edge_hit = bool(
        result[
            "edge_hit"
        ]
    )


    detected_horizontals[
        name
    ] = {

        "position":
            detected_y,

        "edge_hit":
            edge_hit
    }


    offset = (
        detected_y
        - expected_y
    )


    status = (

        "EDGE_HIT"

        if edge_hit

        else "OK"
    )


    print(

        f"{name:20s} "

        f"expected="
        f"{expected_y:7.2f} "

        f"detected="
        f"{detected_y:7.2f} "

        f"offset="
        f"{offset:+7.2f}px "

        f"width="
        f"{stripe_width:2d}px "

        f"score="
        f"{strength:6.1f} "

        f"{status}"
    )


# ============================================================
# DRAW DEBUG IMAGE
# ============================================================

debug = median_background.copy()


# 紅色：
# coarse Homography 理論位置
RED = (
    0,
    0,
    255
)


# 青色：
# CV 找到的白線
CYAN = (
    255,
    255,
    0
)


# 紫色：
# 撞搜尋邊界
MAGENTA = (
    255,
    0,
    255
)


# ============================================================
# DRAW VERTICAL
# ============================================================

for name, x_m in VERTICAL_LINES.items():

    expected_x = meter_to_x(
        x_m
    )


    # 理論線
    if name == "center":

        cv2.line(
            debug,

            (
                expected_x,
                meter_to_y(
                    0.0
                )
            ),

            (
                expected_x,
                meter_to_y(
                    4.72
                )
            ),

            RED,
            1
        )


        cv2.line(
            debug,

            (
                expected_x,
                meter_to_y(
                    8.68
                )
            ),

            (
                expected_x,
                meter_to_y(
                    13.40
                )
            ),

            RED,
            1
        )

    else:

        cv2.line(
            debug,

            (
                expected_x,
                court_top
            ),

            (
                expected_x,
                court_bottom
            ),

            RED,
            1
        )


    if name not in detected_verticals:
        continue


    info = detected_verticals[
        name
    ]


    detected_x = int(
        round(
            info[
                "position"
            ]
        )
    )


    draw_color = (

        MAGENTA

        if info[
            "edge_hit"
        ]

        else CYAN
    )


    if name == "center":

        cv2.line(
            debug,

            (
                detected_x,
                meter_to_y(
                    0.0
                )
            ),

            (
                detected_x,
                meter_to_y(
                    4.72
                )
            ),

            draw_color,
            2
        )


        cv2.line(
            debug,

            (
                detected_x,
                meter_to_y(
                    8.68
                )
            ),

            (
                detected_x,
                meter_to_y(
                    13.40
                )
            ),

            draw_color,
            2
        )


    else:

        cv2.line(
            debug,

            (
                detected_x,
                court_top
            ),

            (
                detected_x,
                court_bottom
            ),

            draw_color,
            2
        )


# ============================================================
# DRAW HORIZONTAL
# ============================================================

for name, y_m in HORIZONTAL_LINES.items():

    expected_y = meter_to_y(
        y_m
    )


    # 理論線
    cv2.line(
        debug,

        (
            court_left,
            expected_y
        ),

        (
            court_right,
            expected_y
        ),

        RED,
        1
    )


    if name not in detected_horizontals:
        continue


    info = detected_horizontals[
        name
    ]


    detected_y = int(
        round(
            info[
                "position"
            ]
        )
    )


    draw_color = (

        MAGENTA

        if info[
            "edge_hit"
        ]

        else CYAN
    )


    cv2.line(
        debug,

        (
            court_left,
            detected_y
        ),

        (
            court_right,
            detected_y
        ),

        draw_color,
        2
    )


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DETECTION
    ),
    debug
)


print()
print(
    "===================================="
)

print(
    "完成"
)

print(
    "===================================="
)

print()
print(
    "Median Background："
)

print(
    OUTPUT_BACKGROUND
)

print()
print(
    "Line Detection："
)

print(
    OUTPUT_DETECTION
)

print()
print(
    "顏色："
)

print(
    "紅線 = AI / Homography 理論位置"
)

print(
    "青線 = CV 白線中心"
)

print(
    "紫線 = 搜尋撞邊界，不可信"
)


# ============================================================
# SHOW
# ============================================================
# ============================================================
# SAVE DETECTED LINE DATA
# ============================================================

line_data = {
    "pixels_per_meter": PIXELS_PER_METER,
    "padding": PADDING,
    "output_width": OUTPUT_WIDTH,
    "output_height": OUTPUT_HEIGHT,

    "vertical_lines": {},

    "horizontal_lines": {}
}


for name, x_m in VERTICAL_LINES.items():

    if name not in detected_verticals:
        continue

    info = detected_verticals[name]

    line_data["vertical_lines"][name] = {

        "court_m": float(x_m),

        "expected_px": float(
            meter_to_x(x_m)
        ),

        "detected_px": float(
            info["position"]
        ),

        "edge_hit": bool(
            info["edge_hit"]
        )
    }


for name, y_m in HORIZONTAL_LINES.items():

    if name not in detected_horizontals:
        continue

    info = detected_horizontals[name]

    line_data["horizontal_lines"][name] = {

        "court_m": float(y_m),

        "expected_px": float(
            meter_to_y(y_m)
        ),

        "detected_px": float(
            info["position"]
        ),

        "edge_hit": bool(
            info["edge_hit"]
        )
    }


OUTPUT_LINES_JSON.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_LINES_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        line_data,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print("Detected lines JSON：")
print(OUTPUT_LINES_JSON)
scale = min(
    900 / OUTPUT_WIDTH,
    900 / OUTPUT_HEIGHT,
    1.0
)


display = cv2.resize(
    debug,

    (
        int(
            OUTPUT_WIDTH
            * scale
        ),

        int(
            OUTPUT_HEIGHT
            * scale
        )
    )
)


cv2.imshow(
    "Bird-eye Court Line Detection V3B",
    display
)


cv2.waitKey(0)

cv2.destroyAllWindows()