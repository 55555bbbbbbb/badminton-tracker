from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "Videos" / "test.mp4"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

OUTPUT_IMAGE = PROJECT_ROOT / "outputs" / "court_multiframe.jpg"
OUTPUT_JSON = PROJECT_ROOT / "calibration" / "auto_court.json"


# ============================================================
# FIND MODEL
# ============================================================

models = list(
    THIRD_PARTY_DIR.rglob("best.pt")
)

if not models:
    raise FileNotFoundError("找不到 best.pt")

MODEL_PATH = models[0]

print("使用模型：")
print(MODEL_PATH)


# ============================================================
# REAL COURT COORDINATES
# ============================================================

COURT_POINTS = np.array(
    [
        [0.00,  0.00],    # K0
        [0.46,  0.00],    # K1
        [3.05,  0.00],    # K2
        [5.64,  0.00],    # K3
        [6.10,  0.00],    # K4

        [0.00,  0.76],    # K5
        [6.10,  0.76],    # K6

        [0.00,  4.72],    # K7
        [3.05,  4.72],    # K8
        [6.10,  4.72],    # K9

        [0.00,  6.70],    # K10
        [6.10,  6.70],    # K11

        [0.00,  8.68],    # K12
        [3.05,  8.68],    # K13
        [6.10,  8.68],    # K14

        [0.00, 12.64],    # K15
        [6.10, 12.64],    # K16

        [0.00, 13.40],    # K17
        [0.46, 13.40],    # K18
        [3.05, 13.40],    # K19
        [5.64, 13.40],    # K20
        [6.10, 13.40],    # K21
    ],
    dtype=np.float32
)


# K10/K11 暫時不參與 Homography
EXCLUDED = {10, 11}

KEYPOINT_CONF = 0.25

# 每個 K 至少要在幾個 frame 出現
MIN_SAMPLES = 3


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# VIDEO INFO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("無法開啟影片")


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = frame_count / fps


print()
print(f"FPS = {fps:.2f}")
print(f"影片長度 = {duration:.2f} 秒")


# ============================================================
# SAMPLE TIMES
#
# 第一版先看 1~8 秒
# ============================================================

sample_times = [
    t for t in
    [1, 2, 3, 4, 5, 6, 7, 8]
    if t < duration
]


print("取樣時間：", sample_times)


# 每個 K 儲存：
# [
#   [x,y,confidence],
#   ...
# ]
observations = {
    k: []
    for k in range(22)
}


reference_frame = None


# ============================================================
# MULTI-FRAME INFERENCE
# ============================================================

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
        print(
            f"{second}s 讀取失敗"
        )
        continue


    if reference_frame is None:
        reference_frame = frame.copy()


    # ============================================================
    # IMPORTANT:
    # ShuttleVision 的 dataset 是 Stretch 到 640x640 訓練
    #
    # 所以 inference 也先模擬相同 preprocessing
    # ============================================================

    original_height, original_width = frame.shape[:2]

    MODEL_SIZE = 640


    # 強制 Stretch 成 640 x 640
    model_frame = cv2.resize(
        frame,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR
    )


    results = model.predict(
        model_frame,

        imgsz=MODEL_SIZE,

        conf=0.15,

        verbose=False
    )   

    result = results[0]


    if (
        result.boxes is None
        or result.keypoints is None
    ):
        print(
            f"{second}s 沒找到結果"
        )
        continue


    class_ids = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    box_conf = (
        result.boxes.conf
        .cpu()
        .numpy()
    )


    court_candidates = []


    for i, class_id in enumerate(class_ids):

        class_name = result.names[
            class_id
        ]

        if class_name == "badminton_court":

            court_candidates.append(
                (
                    i,
                    float(box_conf[i])
                )
            )


    if not court_candidates:

        print(
            f"{second}s 沒找到 Court"
        )

        continue


    # 選 confidence 最大的 Court
    court_candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    court_index = (
        court_candidates[0][0]
    )

    court_conf = (
        court_candidates[0][1]
    )


    xy_model = (
        result.keypoints.xy
        .cpu()
        .numpy()
        [court_index]
    )


    # ============================================================
    # 640x640 座標
    # →
    # 原始影片座標
    # ============================================================

    xy = xy_model.copy()


    scale_x = original_width / MODEL_SIZE
    scale_y = original_height / MODEL_SIZE


    xy[:, 0] *= scale_x
    xy[:, 1] *= scale_y


    conf = (
        result.keypoints.conf
        .cpu()
        .numpy()
        [court_index]
    )


    print(
        f"{second}s "
        f"Court confidence="
        f"{court_conf:.3f}"
    )


    for k in range(22):

        kconf = float(
            conf[k]
        )

        x = float(
            xy[k][0]
        )

        y = float(
            xy[k][1]
        )


        if kconf < KEYPOINT_CONF:
            continue

        if x <= 1 or y <= 1:
            continue


        observations[k].append(
            [
                x,
                y,
                kconf
            ]
        )


cap.release()


if reference_frame is None:
    raise RuntimeError(
        "沒有成功讀到任何 frame"
    )


# ============================================================
# COMPUTE MEDIAN KEYPOINT
# ============================================================

median_points = {}

print()
print(
    "========== MULTI-FRAME KEYPOINTS =========="
)


for k in range(22):

    samples = observations[k]


    if len(samples) < MIN_SAMPLES:

        print(
            f"K{k:02d}: "
            f"SKIP "
            f"samples={len(samples)}"
        )

        continue


    samples_np = np.array(
        samples,
        dtype=np.float32
    )


    median_x = float(
        np.median(samples_np[:, 0])
    )

    median_y = float(
        np.median(samples_np[:, 1])
    )

    median_conf = float(
        np.median(samples_np[:, 2])
    )


    # 看每個點跨 frame 抖多大
    distances = np.sqrt(
        (samples_np[:, 0] - median_x) ** 2
        +
        (samples_np[:, 1] - median_y) ** 2
    )


    median_jitter = float(
        np.median(distances)
    )


    median_points[k] = (
        median_x,
        median_y,
        median_conf,
        median_jitter
    )


    print(
        f"K{k:02d}: "
        f"({median_x:.1f}, {median_y:.1f}) "
        f"samples={len(samples)} "
        f"conf={median_conf:.3f} "
        f"jitter={median_jitter:.2f}px"
    )


# ============================================================
# BUILD HOMOGRAPHY INPUT
# ============================================================

image_points = []
world_points = []
used_indices = []


for k, values in median_points.items():

    if k in EXCLUDED:
        continue


    x, y, conf, jitter = values


    image_points.append(
        [x, y]
    )

    world_points.append(
        COURT_POINTS[k]
    )

    used_indices.append(k)


image_points = np.array(
    image_points,
    dtype=np.float32
)

world_points = np.array(
    world_points,
    dtype=np.float32
)


if len(image_points) < 4:

    raise RuntimeError(
        "不足 4 個有效 keypoints"
    )


# ============================================================
# FIND HOMOGRAPHY
# ============================================================

H_court_to_image, mask = cv2.findHomography(
    world_points,
    image_points,

    method=cv2.RANSAC,

    ransacReprojThreshold=8.0
)


if H_court_to_image is None:
    raise RuntimeError(
        "Homography 計算失敗"
    )


H_image_to_court = np.linalg.inv(
    H_court_to_image
)


mask = mask.ravel().astype(bool)


# ============================================================
# CALCULATE REPROJECTION ERROR
# ============================================================

projected = cv2.perspectiveTransform(
    world_points.reshape(-1, 1, 2),
    H_court_to_image
).reshape(-1, 2)


errors = np.linalg.norm(
    projected - image_points,
    axis=1
)


print()
print(
    "========== HOMOGRAPHY QUALITY =========="
)


inlier_count = 0


for k, error, is_inlier in zip(
    used_indices,
    errors,
    mask
):

    if is_inlier:
        status = "INLIER"
        inlier_count += 1
    else:
        status = "OUTLIER"


    print(
        f"K{k:02d}: "
        f"{status:7s} "
        f"error={error:.2f}px"
    )


print()
print(
    f"RANSAC Inliers: "
    f"{inlier_count} / "
    f"{len(used_indices)}"
)

print(
    f"Median reprojection error: "
    f"{np.median(errors):.2f}px"
)
inlier_errors = errors[mask]


print(
    f"Median ALL error: "
    f"{np.median(errors):.2f}px"
)


if len(inlier_errors) > 0:

    print(
        f"Median INLIER error: "
        f"{np.median(inlier_errors):.2f}px"
    )

    print(
        f"Max INLIER error: "
        f"{np.max(inlier_errors):.2f}px"
    )

# ============================================================
# DRAW HELPERS
# ============================================================

overlay = reference_frame.copy()


def court_to_image(x, y):

    p = np.array(
        [[[x, y]]],
        dtype=np.float32
    )


    q = cv2.perspectiveTransform(
        p,
        H_court_to_image
    )


    return (
        int(round(q[0, 0, 0])),
        int(round(q[0, 0, 1]))
    )


def line(
    x1,
    y1,
    x2,
    y2,
    color=(0, 0, 255),
    thickness=2
):

    cv2.line(
        overlay,

        court_to_image(
            x1,
            y1
        ),

        court_to_image(
            x2,
            y2
        ),

        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW COURT MODEL
# ============================================================

# outer border
line(0, 0, 6.10, 0, thickness=3)
line(6.10, 0, 6.10, 13.40, thickness=3)
line(6.10, 13.40, 0, 13.40, thickness=3)
line(0, 13.40, 0, 0, thickness=3)

# singles sidelines
line(0.46, 0, 0.46, 13.40)
line(5.64, 0, 5.64, 13.40)

# doubles long service
line(0, 0.76, 6.10, 0.76)
line(0, 12.64, 6.10, 12.64)

# short service
line(0, 4.72, 6.10, 4.72)
line(0, 8.68, 6.10, 8.68)

# centre
line(3.05, 0, 3.05, 4.72)
line(3.05, 8.68, 3.05, 13.40)

# net
line(
    0,
    6.70,
    6.10,
    6.70,
    color=(255, 0, 255),
    thickness=3
)


# ============================================================
# DRAW MEDIAN KEYPOINTS
#
# GREEN = RANSAC inlier
# RED   = RANSAC outlier
# ============================================================

for index, k in enumerate(
    used_indices
):

    x, y, conf, jitter = (
        median_points[k]
    )


    if mask[index]:
        color = (
            0,
            255,
            0
        )
    else:
        color = (
            0,
            0,
            255
        )


    cv2.circle(
        overlay,
        (
            int(round(x)),
            int(round(y))
        ),
        7,
        color,
        -1
    )


    cv2.putText(
        overlay,

        f"K{k}",

        (
            int(round(x)) + 8,
            int(round(y)) - 8
        ),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE JSON
# ============================================================

OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True
)


data = {
    "video": VIDEO_PATH.name,

    "sample_times": sample_times,

    "used_keypoints": used_indices,

    "ransac_inliers": [
        int(k)
        for k, ok in zip(
            used_indices,
            mask
        )
        if ok
    ],

    "median_reprojection_error_px":
        float(np.median(errors)),

    "homography_court_to_image":
        H_court_to_image.tolist(),

    "homography_image_to_court":
        H_image_to_court.tolist(),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SAVE IMAGE
# ============================================================

result_image = cv2.addWeighted(
    overlay,
    0.72,
    reference_frame,
    0.28,
    0
)


OUTPUT_IMAGE.parent.mkdir(
    parents=True,
    exist_ok=True
)


cv2.imwrite(
    str(OUTPUT_IMAGE),
    result_image
)


print()
print("輸出：")
print(OUTPUT_IMAGE)
print(OUTPUT_JSON)


# ============================================================
# SHOW
# ============================================================

h, w = result_image.shape[:2]

scale = min(
    1400 / w,
    850 / h,
    1.0
)


display = cv2.resize(
    result_image,
    (
        int(w * scale),
        int(h * scale)
    )
)


cv2.imshow(
    "Court Multi Frame",
    display
)

cv2.waitKey(0)

cv2.destroyAllWindows()