from pathlib import Path
import argparse
import csv
import json
import math

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# SETTINGS
# ============================================================

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
NET_Y_M = 6.70

PERSON_CONF = 0.20
ANKLE_CONF = 0.20
COURT_MARGIN_M = 1.00
IMGSZ = 960

IDENTITY_TIMEOUT_FRAMES = 150
PREDICT_DISPLAY_FRAMES = 12
SIDE_TOLERANCE_M = 0.65
MAX_MATCH_DISTANCE_M = 3.20
RAW_ID_BONUS_M = 0.65

EMA_ALPHA = 0.45
VELOCITY_ALPHA = 0.35
MAX_SPEED_M_PER_FRAME = 0.30
SMOOTH_RESET_GAP_FRAMES = 12

MAP_WIDTH = 240
MAP_HEIGHT = 480
MAP_PADDING = 20


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--video",
    required=True,
    help="輸入影片"
)

parser.add_argument(
    "--calibration",
    required=True,
    help="對應影片的 refined_court_v2.json"
)

parser.add_argument(
    "--match-format",
    required=True,
    choices=("singles", "doubles"),
    help=(
        "比賽型態。singles 只允許 P1/P3；"
        "doubles 允許 P1/P2/P3/P4"
    )
)

parser.add_argument(
    "--start",
    type=float,
    default=0.0,
    help="從第幾秒開始"
)

parser.add_argument(
    "--duration",
    type=float,
    default=None,
    help="分析幾秒；不填就跑完整影片"
)

parser.add_argument(
    "--debug-tracks",
    action="store_true",
    help=(
        "額外輸出 tracker 原始人物結果；"
        "用來區分 detector/tracker miss 與 court filter reject"
    )
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

video_path = Path(args.video)

if not video_path.is_absolute():
    video_path = PROJECT_ROOT / video_path

video_path = video_path.resolve()

calibration_path = Path(args.calibration)

if not calibration_path.is_absolute():
    calibration_path = PROJECT_ROOT / calibration_path

calibration_path = calibration_path.resolve()

if not video_path.exists():
    raise FileNotFoundError(
        f"找不到影片：{video_path}"
    )

if not calibration_path.exists():
    raise FileNotFoundError(
        f"找不到 Calibration：{calibration_path}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

video_group = video_path.parent.name

output_video_path = (
    OUTPUT_DIR
    / f"{video_group}_{video_path.stem}_identity.mp4"
)

output_csv_path = (
    OUTPUT_DIR
    / f"{video_group}_{video_path.stem}_identity_positions.csv"
)

debug_tracks_csv_path = (
    OUTPUT_DIR
    / f"{video_group}_{video_path.stem}_identity_raw_tracks.csv"
)


# ============================================================
# MODEL
# ============================================================

pose_models = list(
    THIRD_PARTY_DIR.rglob(
        "yolov8m-pose.pt"
    )
)

if not pose_models:
    raise FileNotFoundError(
        "third_party 裡找不到 yolov8m-pose.pt"
    )

MODEL_PATH = pose_models[0]


# ============================================================
# CALIBRATION
# ============================================================

with open(
    calibration_path,
    "r",
    encoding="utf-8"
) as f:
    calibration = json.load(f)

H_image_to_court = np.array(
    calibration["homography_image_to_court"],
    dtype=np.float64
)

H_court_to_image = np.array(
    calibration["homography_court_to_image"],
    dtype=np.float64
)


def image_to_court(x, y):

    p = np.array(
        [[[
            float(x),
            float(y)
        ]]],
        dtype=np.float32
    )

    q = cv2.perspectiveTransform(
        p,
        H_image_to_court
    )

    return (
        float(q[0, 0, 0]),
        float(q[0, 0, 1])
    )


def court_to_image(x, y):

    p = np.array(
        [[[
            float(x),
            float(y)
        ]]],
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


# ============================================================
# COLORS
# ============================================================

COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (0, 220, 255),
]


def player_color(player_id):

    return COLORS[
        (int(player_id) - 1)
        %
        len(COLORS)
    ]


# ============================================================
# GROUND POINT
# ============================================================

def get_ground_point(
    box,
    keypoints,
    confidences
):

    x1, y1, x2, y2 = box

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if confidences is not None:

        left_conf = float(
            confidences[15]
        )

        right_conf = float(
            confidences[16]
        )

    else:

        left_conf = 1.0
        right_conf = 1.0

    left_valid = (
        left_conf >= ANKLE_CONF
        and left_ankle[0] > 1
        and left_ankle[1] > 1
    )

    right_valid = (
        right_conf >= ANKLE_CONF
        and right_ankle[0] > 1
        and right_ankle[1] > 1
    )

    if left_valid and right_valid:

        ground_x = float(
            (
                left_ankle[0]
                +
                right_ankle[0]
            )
            /
            2
        )

        ground_y = float(
            (
                left_ankle[1]
                +
                right_ankle[1]
            )
            /
            2
        )

        method = "ankle_midpoint"

    elif left_valid:

        ground_x = float(
            left_ankle[0]
        )

        ground_y = float(
            left_ankle[1]
        )

        method = "left_ankle"

    elif right_valid:

        ground_x = float(
            right_ankle[0]
        )

        ground_y = float(
            right_ankle[1]
        )

        method = "right_ankle"

    else:

        ground_x = float(
            (x1 + x2) / 2
        )

        ground_y = float(
            y2
        )

        method = "bbox_bottom"

    return {
        "x": ground_x,
        "y": ground_y,
        "method": method,
        "left_conf": left_conf,
        "right_conf": right_conf,
    }


# ============================================================
# LOGICAL PLAYER STATE
# ============================================================

class PlayerState:

    def __init__(
        self,
        player_id,
        side
    ):
        self.player_id = (
            player_id
        )

        self.side = side

        self.active = False

        self.last_seen_frame = (
            -10**9
        )

        self.last_raw_track_id = (
            None
        )

        self.raw_x = None
        self.raw_y = None

        # Ground Position Correction V1:
        # corrected_* 是最後可信的 court position；raw_* 永遠保留最新量測。
        self.corrected_x = None
        self.corrected_y = None
        self.last_position_frame = -10**9

        self.stable_x = None
        self.stable_y = None

        self.vx = 0.0
        self.vy = 0.0

    def missing_frames(
        self,
        frame_index
    ):

        if not self.active:
            return 10**9

        return max(
            0,
            frame_index
            -
            self.last_seen_frame
        )

    def predict(
        self,
        frame_index
    ):

        if not self.active:
            return None

        if (
            self.corrected_x is None
            or
            self.corrected_y is None
        ):
            return None

        # Position prediction advances from the last accepted ground point.
        # A rejected ground measurement still counts as "seen" for identity,
        # but it must not become the new motion origin.
        dt = max(
            0,
            frame_index
            -
            self.last_position_frame
        )

        return (
            self.corrected_x
            +
            self.vx
            *
            dt,

            self.corrected_y
            +
            self.vy
            *
            dt
        )

    def reset(self):

        self.active = False

        self.last_seen_frame = (
            -10**9
        )

        self.last_raw_track_id = (
            None
        )

        self.raw_x = None
        self.raw_y = None

        # Ground Position Correction V1:
        # corrected_* 是最後可信的 court position；raw_* 永遠保留最新量測。
        self.corrected_x = None
        self.corrected_y = None
        self.last_position_frame = -10**9

        self.stable_x = None
        self.stable_y = None

        self.vx = 0.0
        self.vy = 0.0

    def update(
        self,
        detection,
        frame_index
    ):

        new_x = float(
            detection["raw_x"]
        )

        new_y = float(
            detection["raw_y"]
        )

        ground_method = detection.get(
            "ground_method",
            "unknown"
        )

        if ground_method == "ankle_midpoint":
            ground_quality = "two_ankles"
        elif ground_method in (
            "left_ankle",
            "right_ankle"
        ):
            ground_quality = "single_ankle"
        else:
            ground_quality = "bbox_fallback"

        ground_status = "accepted"
        ground_speed = None

        if not self.active:

            self.active = True

            self.raw_x = new_x
            self.raw_y = new_y

            self.corrected_x = new_x
            self.corrected_y = new_y
            self.last_position_frame = frame_index

            self.stable_x = new_x
            self.stable_y = new_y

            self.vx = 0.0
            self.vy = 0.0

            corrected_x = new_x
            corrected_y = new_y

        else:

            dt_position = max(
                1,
                frame_index
                -
                self.last_position_frame
            )

            position_dx = (
                new_x
                -
                self.corrected_x
            )

            position_dy = (
                new_y
                -
                self.corrected_y
            )

            ground_speed = (
                math.hypot(
                    position_dx,
                    position_dy
                )
                /
                dt_position
            )

            # Reuse the project's existing physical motion bound.  Previously
            # it only clipped velocity; V1 also prevents a short-gap raw ground
            # spike from contaminating corrected/stable XY.
            reject_motion_outlier = (
                dt_position
                <=
                SMOOTH_RESET_GAP_FRAMES
                and
                ground_speed
                >
                MAX_SPEED_M_PER_FRAME
            )

            if reject_motion_outlier:

                ground_status = (
                    "corrected_motion_outlier"
                )

                predicted = self.predict(
                    frame_index
                )

                if predicted is None:
                    corrected_x = self.corrected_x
                    corrected_y = self.corrected_y
                else:
                    corrected_x = float(
                        predicted[0]
                    )
                    corrected_y = float(
                        predicted[1]
                    )

                # Keep identity/raw observation, but do not learn motion from
                # the bad ground point.  Stable XY follows the short prediction.
                self.stable_x = (
                    EMA_ALPHA
                    *
                    corrected_x
                    +
                    (1.0 - EMA_ALPHA)
                    *
                    self.stable_x
                )

                self.stable_y = (
                    EMA_ALPHA
                    *
                    corrected_y
                    +
                    (1.0 - EMA_ALPHA)
                    *
                    self.stable_y
                )

            else:

                observed_vx = (
                    position_dx
                    /
                    dt_position
                )

                observed_vy = (
                    position_dy
                    /
                    dt_position
                )

                speed = math.hypot(
                    observed_vx,
                    observed_vy
                )

                if (
                    speed
                    >
                    MAX_SPEED_M_PER_FRAME
                ):

                    scale = (
                        MAX_SPEED_M_PER_FRAME
                        /
                        speed
                    )

                    observed_vx *= scale
                    observed_vy *= scale

                self.vx = (
                    VELOCITY_ALPHA
                    *
                    observed_vx
                    +
                    (1.0 - VELOCITY_ALPHA)
                    *
                    self.vx
                )

                self.vy = (
                    VELOCITY_ALPHA
                    *
                    observed_vy
                    +
                    (1.0 - VELOCITY_ALPHA)
                    *
                    self.vy
                )

                corrected_x = new_x
                corrected_y = new_y

                if (
                    dt_position
                    >
                    SMOOTH_RESET_GAP_FRAMES
                ):

                    self.stable_x = new_x
                    self.stable_y = new_y

                else:

                    self.stable_x = (
                        EMA_ALPHA
                        *
                        new_x
                        +
                        (1.0 - EMA_ALPHA)
                        *
                        self.stable_x
                    )

                    self.stable_y = (
                        EMA_ALPHA
                        *
                        new_y
                        +
                        (1.0 - EMA_ALPHA)
                        *
                        self.stable_y
                    )

                self.corrected_x = new_x
                self.corrected_y = new_y
                self.last_position_frame = frame_index

            # Raw measurement is always retained, even if rejected for stable XY.
            self.raw_x = new_x
            self.raw_y = new_y

        self.last_seen_frame = frame_index

        self.last_raw_track_id = detection[
            "raw_track_id"
        ]

        return {
            "ground_quality": ground_quality,
            "ground_status": ground_status,
            "corrected_x": corrected_x,
            "corrected_y": corrected_y,
            "ground_speed_m_per_frame": ground_speed,
        }


# ============================================================
# PERSISTENT IDENTITY MANAGER
#
# P1/P2 = 遠半場
# P3/P4 = 近半場
# ============================================================

class IdentityManager:

    def __init__(
        self,
        match_format
    ):

        if match_format not in (
            "singles",
            "doubles"
        ):
            raise ValueError(
                f"不支援的 match format：{match_format}"
            )

        self.match_format = match_format

        self.states = {
            1: PlayerState(
                1,
                "far"
            ),
            2: PlayerState(
                2,
                "far"
            ),
            3: PlayerState(
                3,
                "near"
            ),
            4: PlayerState(
                4,
                "near"
            ),
        }

        if match_format == "singles":
            self.allowed_player_ids = {
                1,
                3,
            }
        else:
            self.allowed_player_ids = {
                1,
                2,
                3,
                4,
            }

    def allowed_states(
        self
    ):

        return [
            state
            for player_id, state
            in self.states.items()
            if player_id
            in self.allowed_player_ids
        ]

    def raw_id_owner(
        self,
        raw_track_id
    ):

        owners = [
            state
            for state
            in self.allowed_states()
            if (
                state.active
                and
                state.last_raw_track_id
                ==
                raw_track_id
            )
        ]

        if len(owners) == 1:
            return owners[0]

        # 理論上 active raw ID 應只有一個 owner。
        # 若真的重複，視為 ambiguous，任何 state 都不能搶。
        if len(owners) > 1:
            return False

        return None

    def ownership_compatible(
        self,
        state,
        detection
    ):

        owner = self.raw_id_owner(
            detection[
                "raw_track_id"
            ]
        )

        if owner is None:
            return True

        if owner is False:
            return False

        return (
            owner.player_id
            ==
            state.player_id
        )

    def detection_has_active_owner(
        self,
        detection
    ):

        owner = self.raw_id_owner(
            detection[
                "raw_track_id"
            ]
        )

        return (
            owner is not None
        )

    def side_compatible(
        self,
        side,
        detection
    ):

        y = detection[
            "raw_y"
        ]

        if side == "far":

            return (
                y
                <=
                NET_Y_M
                +
                SIDE_TOLERANCE_M
            )

        return (
            y
            >=
            NET_Y_M
            -
            SIDE_TOLERANCE_M
        )

    def match_threshold(
        self,
        state,
        frame_index
    ):

        missing = min(
            state.missing_frames(
                frame_index
            ),
            30
        )

        return min(
            MAX_MATCH_DISTANCE_M,
            1.00
            +
            0.07
            *
            missing
        )

    def pair_cost(
        self,
        state,
        detection,
        frame_index
    ):

        # Raw ByteTrack ID ownership protection:
        # 只要某個 raw ID 仍屬於 active logical player，
        # 其他 P# 不得把它搶走。位置/side 衝突時寧可暫時 predicted。
        if not self.ownership_compatible(
            state,
            detection
        ):
            return None

        if not self.side_compatible(
            state.side,
            detection
        ):

            return None

        predicted = state.predict(
            frame_index
        )

        if predicted is None:
            return None

        px, py = predicted

        distance = math.hypot(
            detection["raw_x"]
            -
            px,

            detection["raw_y"]
            -
            py
        )

        threshold = self.match_threshold(
            state,
            frame_index
        )

        if (
            state.last_raw_track_id
            ==
            detection["raw_track_id"]
        ):

            threshold = max(
                threshold,
                2.25
            )

            cost = max(
                0.0,
                distance
                -
                RAW_ID_BONUS_M
            )

        else:

            cost = distance

        if distance > threshold:
            return None

        return cost

    def best_matching(
        self,
        states,
        detections,
        frame_index
    ):

        best_pairs = []
        best_match_count = -1
        best_cost = float(
            "inf"
        )

        def search(
            state_index,
            used_detection_indices,
            pairs,
            total_cost
        ):

            nonlocal best_pairs
            nonlocal best_match_count
            nonlocal best_cost

            if (
                state_index
                >=
                len(states)
            ):

                count = len(
                    pairs
                )

                if (
                    count
                    >
                    best_match_count
                    or
                    (
                        count
                        ==
                        best_match_count
                        and
                        total_cost
                        <
                        best_cost
                    )
                ):

                    best_match_count = (
                        count
                    )

                    best_cost = (
                        total_cost
                    )

                    best_pairs = list(
                        pairs
                    )

                return

            state = states[
                state_index
            ]

            # skip
            search(
                state_index + 1,
                used_detection_indices,
                pairs,
                total_cost
            )

            # match
            for (
                det_index,
                detection
            ) in enumerate(
                detections
            ):

                if (
                    det_index
                    in
                    used_detection_indices
                ):
                    continue

                cost = self.pair_cost(
                    state,
                    detection,
                    frame_index
                )

                if cost is None:
                    continue

                used_detection_indices.add(
                    det_index
                )

                pairs.append(
                    (
                        state,
                        det_index,
                        cost
                    )
                )

                search(
                    state_index + 1,
                    used_detection_indices,
                    pairs,
                    total_cost
                    +
                    cost
                )

                pairs.pop()

                used_detection_indices.remove(
                    det_index
                )

        search(
            0,
            set(),
            [],
            0.0
        )

        return best_pairs

    def deactivate_stale(
        self,
        frame_index
    ):

        for state in (
            self.allowed_states()
        ):

            if (
                state.active
                and
                state.missing_frames(
                    frame_index
                )
                >
                IDENTITY_TIMEOUT_FRAMES
            ):

                state.reset()

    def assign(
        self,
        detections,
        frame_index
    ):

        self.deactivate_stale(
            frame_index
        )

        assignments = []

        used_detection_uids = (
            set()
        )

        # Existing IDs first
        for side in (
            "far",
            "near"
        ):

            active_states = [
                state
                for state
                in self.allowed_states()
                if (
                    state.active
                    and
                    state.side
                    ==
                    side
                )
            ]

            candidate_detections = [
                detection
                for detection
                in detections
                if self.side_compatible(
                    side,
                    detection
                )
            ]

            pairs = self.best_matching(
                active_states,
                candidate_detections,
                frame_index
            )

            for (
                state,
                local_det_index,
                cost
            ) in pairs:

                detection = (
                    candidate_detections[
                        local_det_index
                    ]
                )

                det_uid = detection[
                    "uid"
                ]

                if (
                    det_uid
                    in
                    used_detection_uids
                ):
                    continue

                position_info = state.update(
                    detection,
                    frame_index
                )

                used_detection_uids.add(
                    det_uid
                )

                assignments.append(
                    {
                        "player_id":
                            state.player_id,

                        "detection":
                            detection,

                        "match_cost":
                            cost,

                        "match_mode":
                            "strict",

                        "position_info":
                            position_info,
                    }
                )

        # Relaxed reacquire:
        # strict matching 後，同側若只剩 1 個 active-but-unmatched
        # logical state + 1 個 unmatched detection，直接接回原 P#。
        # 2 x 2 以上維持 ambiguous，不強制配對，避免 swap。
        assigned_player_ids = {
            item["player_id"]
            for item
            in assignments
        }

        for side in (
            "far",
            "near"
        ):

            unmatched_states = [
                state
                for state
                in self.allowed_states()
                if (
                    state.active
                    and
                    state.side
                    ==
                    side
                    and
                    state.player_id
                    not in
                    assigned_player_ids
                )
            ]

            if (
                len(unmatched_states)
                !=
                1
            ):
                continue

            state = unmatched_states[0]

            unmatched_detections = [
                detection
                for detection
                in detections
                if (
                    detection["uid"]
                    not in
                    used_detection_uids
                    and
                    self.side_compatible(
                        side,
                        detection
                    )
                    and
                    self.ownership_compatible(
                        state,
                        detection
                    )
                )
            ]

            if (
                len(unmatched_detections)
                !=
                1
            ):
                continue

            detection = unmatched_detections[0]

            predicted_position = state.predict(
                frame_index
            )

            if predicted_position is None:
                continue

            match_cost = math.hypot(
                detection["raw_x"]
                -
                predicted_position[0],
                detection["raw_y"]
                -
                predicted_position[1]
            )

            position_info = state.update(
                detection,
                frame_index
            )

            used_detection_uids.add(
                detection["uid"]
            )

            assigned_player_ids.add(
                state.player_id
            )

            assignments.append(
                {
                    "player_id":
                        state.player_id,

                    "detection":
                        detection,

                    "match_cost":
                        match_cost,

                    "match_mode":
                        "relaxed_reacquire",

                    "position_info":
                        position_info,
                }
            )

        # New logical players if a side still has an unused slot.
        # 重要：只要同側仍有 active-but-unmatched state，先不要建立新 P#。
        # 這避免單打球員短暫 lost / 出界後回來時，被誤建成 P2 / P4。
        unmatched = [
            detection
            for detection
            in detections
            if (
                detection["uid"]
                not in
                used_detection_uids
            )
        ]

        for side in (
            "far",
            "near"
        ):

            assigned_player_ids = {
                item["player_id"]
                for item
                in assignments
            }

            has_unmatched_active_state = any(
                state.active
                and
                state.side
                ==
                side
                and
                state.player_id
                not in
                assigned_player_ids
                for state
                in self.allowed_states()
            )

            if has_unmatched_active_state:
                continue

            side_detections = [
                detection
                for detection
                in unmatched
                if (
                    self.side_compatible(
                        side,
                        detection
                    )
                    and
                    not self.detection_has_active_owner(
                        detection
                    )
                )
            ]

            side_detections.sort(
                key=lambda d:
                d["person_conf"],
                reverse=True
            )

            inactive_states = [
                state
                for state
                in self.allowed_states()
                if (
                    not state.active
                    and
                    state.side
                    ==
                    side
                )
            ]

            inactive_states.sort(
                key=lambda state:
                state.player_id
            )

            side_detections = (
                side_detections[
                    :len(
                        inactive_states
                    )
                ]
            )

            side_detections.sort(
                key=lambda detection:
                detection["raw_x"]
            )

            for (
                state,
                detection
            ) in zip(
                inactive_states,
                side_detections
            ):

                position_info = state.update(
                    detection,
                    frame_index
                )

                used_detection_uids.add(
                    detection["uid"]
                )

                assignments.append(
                    {
                        "player_id":
                            state.player_id,

                        "detection":
                            detection,

                        "match_cost":
                            0.0,

                        "match_mode":
                            "new_player",

                        "position_info":
                            position_info,
                    }
                )

        # Short missing periods: keep logical player alive on map
        visible_ids = {
            item["player_id"]
            for item
            in assignments
        }

        predicted = []

        for state in (
            self.allowed_states()
        ):

            if not state.active:
                continue

            if (
                state.player_id
                in
                visible_ids
            ):
                continue

            missing = (
                state.missing_frames(
                    frame_index
                )
            )

            if (
                missing
                <=
                PREDICT_DISPLAY_FRAMES
            ):

                pred = state.predict(
                    frame_index
                )

                if pred is not None:

                    predicted.append(
                        {
                            "player_id":
                                state.player_id,

                            "pred_x":
                                pred[0],

                            "pred_y":
                                pred[1],

                            "missing_frames":
                                missing,
                        }
                    )

        assignments.sort(
            key=lambda item:
            item["player_id"]
        )

        predicted.sort(
            key=lambda item:
            item["player_id"]
        )

        return (
            assignments,
            predicted
        )


# ============================================================
# MINI MAP
# ============================================================

def create_mini_map(
    assignments,
    predicted,
    identity_manager
):

    mini_map = np.zeros(
        (
            MAP_HEIGHT,
            MAP_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    mini_map[:] = (
        45,
        110,
        45
    )

    usable_w = (
        MAP_WIDTH
        -
        MAP_PADDING
        *
        2
    )

    usable_h = (
        MAP_HEIGHT
        -
        MAP_PADDING
        *
        2
    )

    def point(
        x_m,
        y_m
    ):

        x_display = np.clip(
            x_m,
            0,
            COURT_WIDTH_M
        )

        y_display = np.clip(
            y_m,
            0,
            COURT_LENGTH_M
        )

        px = int(
            round(
                MAP_PADDING
                +
                x_display
                /
                COURT_WIDTH_M
                *
                usable_w
            )
        )

        py = int(
            round(
                MAP_PADDING
                +
                y_display
                /
                COURT_LENGTH_M
                *
                usable_h
            )
        )

        return (
            px,
            py
        )

    def line(
        x1,
        y1,
        x2,
        y2,
        thickness=1
    ):

        cv2.line(
            mini_map,
            point(
                x1,
                y1
            ),
            point(
                x2,
                y2
            ),
            (
                255,
                255,
                255
            ),
            thickness,
            cv2.LINE_AA
        )

    # Court
    line(
        0,
        0,
        6.10,
        0,
        2
    )

    line(
        6.10,
        0,
        6.10,
        13.40,
        2
    )

    line(
        6.10,
        13.40,
        0,
        13.40,
        2
    )

    line(
        0,
        13.40,
        0,
        0,
        2
    )

    line(
        0.46,
        0,
        0.46,
        13.40
    )

    line(
        5.64,
        0,
        5.64,
        13.40
    )

    line(
        0,
        0.76,
        6.10,
        0.76
    )

    line(
        0,
        12.64,
        6.10,
        12.64
    )

    line(
        0,
        4.72,
        6.10,
        4.72
    )

    line(
        0,
        8.68,
        6.10,
        8.68
    )

    line(
        0,
        6.70,
        6.10,
        6.70,
        2
    )

    line(
        3.05,
        0,
        3.05,
        4.72
    )

    line(
        3.05,
        8.68,
        3.05,
        13.40
    )

    for item in assignments:

        player_id = item[
            "player_id"
        ]

        state = (
            identity_manager
            .states[
                player_id
            ]
        )

        px, py = point(
            state.stable_x,
            state.stable_y
        )

        color = player_color(
            player_id
        )

        cv2.circle(
            mini_map,
            (
                px,
                py
            ),
            9,
            color,
            -1
        )

        cv2.putText(
            mini_map,
            f"P{player_id}",
            (
                px + 9,
                py - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    for item in predicted:

        player_id = item[
            "player_id"
        ]

        px, py = point(
            item["pred_x"],
            item["pred_y"]
        )

        color = player_color(
            player_id
        )

        # Hollow dot = predicted, not directly detected
        cv2.circle(
            mini_map,
            (
                px,
                py
            ),
            8,
            color,
            2
        )

        cv2.putText(
            mini_map,
            f"P{player_id}*",
            (
                px + 9,
                py - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA
        )

    return mini_map


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(
        video_path
    )
)

if not cap.isOpened():
    raise RuntimeError(
        "無法開啟影片"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

start_frame = int(
    round(
        args.start
        *
        fps
    )
)

if args.duration is None:

    end_frame = (
        total_frames
    )

else:

    end_frame = min(
        total_frames,
        start_frame
        +
        int(
            round(
                args.duration
                *
                fps
            )
        )
    )

frames_to_process = max(
    0,
    end_frame
    -
    start_frame
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(
        output_video_path
    ),
    fourcc,
    fps,
    (
        width,
        height
    )
)

if not writer.isOpened():
    raise RuntimeError(
        "無法建立輸出 MP4"
    )


# ============================================================
# CSV
# ============================================================

csv_file = open(
    output_csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow(
    [
        "frame",
        "time_sec",
        "status",
        "player_id",
        "raw_track_id",
        "person_conf",
        "ground_method",
        "ground_pixel_x",
        "ground_pixel_y",
        "left_ankle_conf",
        "right_ankle_conf",
        "raw_x_m",
        "raw_y_m",
        "stable_x_m",
        "stable_y_m",
        "match_cost",
        "missing_frames",
        "ground_quality",
        "ground_status",
        "corrected_x_m",
        "corrected_y_m",
        "ground_speed_m_per_frame",
    ]
)

debug_tracks_file = None
debug_tracks_writer = None

if args.debug_tracks:
    debug_tracks_file = open(
        debug_tracks_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    debug_tracks_writer = csv.writer(
        debug_tracks_file
    )

    debug_tracks_writer.writerow(
        [
            "frame",
            "time_sec",
            "raw_track_id",
            "person_conf",
            "ground_method",
            "ground_pixel_x",
            "ground_pixel_y",
            "left_ankle_conf",
            "right_ankle_conf",
            "raw_x_m",
            "raw_y_m",
            "inside_analysis_area",
        ]
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "========================================"
)

print(
    "PLAYER V2.2 - ROSTER-AWARE PERSISTENT IDENTITY"
)

print(
    "========================================"
)

print()
print(
    "Video:"
)
print(
    video_path
)

print()
print(
    "Calibration:"
)
print(
    calibration_path
)

print()
print(
    "Pose model:"
)
print(
    MODEL_PATH
)

print()
print(
    f"Resolution: "
    f"{width} x {height}"
)

print(
    f"FPS: "
    f"{fps:.2f}"
)

print(
    f"Frames: "
    f"{frames_to_process}"
)

print()
print(
    f"Match format: {args.match_format}"
)

print()
print(
    "Logical IDs:"
)

if args.match_format == "singles":
    print(
        "P1 = far court"
    )

    print(
        "P3 = near court"
    )

else:
    print(
        "P1/P2 = far court"
    )

    print(
        "P3/P4 = near court"
    )

print()
print(
    "本版不處理剪接/畫面閃入。"
)

model = YOLO(
    str(
        MODEL_PATH
    )
)

identity_manager = (
    IdentityManager(
        args.match_format
    )
)


# ============================================================
# COURT POLYGON
# ============================================================

court_polygon = np.array(
    [
        court_to_image(
            0,
            0
        ),
        court_to_image(
            6.10,
            0
        ),
        court_to_image(
            6.10,
            13.40
        ),
        court_to_image(
            0,
            13.40
        ),
    ],
    dtype=np.int32
)


# ============================================================
# PROCESS
# ============================================================

processed = 0
frame_index = start_frame

while (
    frame_index
    <
    end_frame
):

    ok, frame = cap.read()

    if not ok:
        break

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=IMGSZ,
        conf=PERSON_CONF,
        classes=[0],
        verbose=False
    )

    result = results[0]

    display = (
        frame.copy()
    )

    cv2.polylines(
        display,
        [
            court_polygon
        ],
        True,
        (
            0,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    detections = []

    if (
        result.boxes
        is not None
        and
        result.keypoints
        is not None
        and
        result.boxes.id
        is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        box_conf = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        raw_track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        keypoint_xy = (
            result.keypoints.xy
            .cpu()
            .numpy()
        )

        if (
            result.keypoints.conf
            is not None
        ):

            keypoint_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

        else:

            keypoint_conf = None

        for i in range(
            len(boxes)
        ):

            confidences = (
                keypoint_conf[i]
                if
                keypoint_conf
                is not None
                else
                None
            )

            ground = get_ground_point(
                boxes[i],
                keypoint_xy[i],
                confidences
            )

            raw_x, raw_y = (
                image_to_court(
                    ground["x"],
                    ground["y"]
                )
            )

            inside_analysis_area = (
                -COURT_MARGIN_M
                <=
                raw_x
                <=
                COURT_WIDTH_M
                +
                COURT_MARGIN_M

                and

                -COURT_MARGIN_M
                <=
                raw_y
                <=
                COURT_LENGTH_M
                +
                COURT_MARGIN_M
            )

            if debug_tracks_writer is not None:
                debug_tracks_writer.writerow(
                    [
                        frame_index,
                        frame_index / fps,
                        int(raw_track_ids[i]),
                        float(box_conf[i]),
                        ground["method"],
                        ground["x"],
                        ground["y"],
                        ground["left_conf"],
                        ground["right_conf"],
                        raw_x,
                        raw_y,
                        int(inside_analysis_area),
                    ]
                )

            if not inside_analysis_area:
                continue

            detections.append(
                {
                    "uid":
                        i,

                    "raw_track_id":
                        int(
                            raw_track_ids[i]
                        ),

                    "box":
                        boxes[i],

                    "person_conf":
                        float(
                            box_conf[i]
                        ),

                    "ground_x":
                        ground["x"],

                    "ground_y":
                        ground["y"],

                    "ground_method":
                        ground["method"],

                    "left_conf":
                        ground["left_conf"],

                    "right_conf":
                        ground["right_conf"],

                    "raw_x":
                        raw_x,

                    "raw_y":
                        raw_y,
                }
            )

    assignments, predicted = (
        identity_manager.assign(
            detections,
            frame_index
        )
    )

    # ========================================================
    # DRAW DETECTED PLAYERS
    # ========================================================

    for item in assignments:

        player_id = item[
            "player_id"
        ]

        detection = item[
            "detection"
        ]

        position_info = item[
            "position_info"
        ]

        state = (
            identity_manager
            .states[
                player_id
            ]
        )

        color = player_color(
            player_id
        )

        (
            x1,
            y1,
            x2,
            y2
        ) = detection["box"]

        cv2.rectangle(
            display,
            (
                int(
                    round(
                        x1
                    )
                ),
                int(
                    round(
                        y1
                    )
                )
            ),
            (
                int(
                    round(
                        x2
                    )
                ),
                int(
                    round(
                        y2
                    )
                )
            ),
            color,
            2
        )

        cv2.circle(
            display,
            (
                int(
                    round(
                        detection[
                            "ground_x"
                        ]
                    )
                ),
                int(
                    round(
                        detection[
                            "ground_y"
                        ]
                    )
                )
            ),
            6,
            color,
            -1
        )

        ground_flag = (
            " G!"
            if
            position_info["ground_status"]
            !=
            "accepted"
            else
            ""
        )

        label = (
            f"P{player_id} "
            f"r{detection['raw_track_id']} "
            f"({state.stable_x:.2f}, "
            f"{state.stable_y:.2f})"
            f"{ground_flag}"
        )

        cv2.putText(
            display,
            label,
            (
                int(
                    round(
                        x1
                    )
                ),
                max(
                    22,
                    int(
                        round(
                            y1
                        )
                    )
                    -
                    8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA
        )

        csv_writer.writerow(
            [
                frame_index,
                frame_index
                /
                fps,
                "detected",

                player_id,

                detection[
                    "raw_track_id"
                ],

                detection[
                    "person_conf"
                ],

                detection[
                    "ground_method"
                ],

                detection[
                    "ground_x"
                ],

                detection[
                    "ground_y"
                ],

                detection[
                    "left_conf"
                ],

                detection[
                    "right_conf"
                ],

                detection[
                    "raw_x"
                ],

                detection[
                    "raw_y"
                ],

                state.stable_x,
                state.stable_y,

                item[
                    "match_cost"
                ],

                0,

                position_info["ground_quality"],
                position_info["ground_status"],
                position_info["corrected_x"],
                position_info["corrected_y"],
                position_info["ground_speed_m_per_frame"],
            ]
        )

    # ========================================================
    # CSV FOR PREDICTED SHORT GAPS
    # ========================================================

    for item in predicted:

        player_id = item[
            "player_id"
        ]

        csv_writer.writerow(
            [
                frame_index,
                frame_index
                /
                fps,
                "predicted",

                player_id,

                "",

                "",
                "motion_prediction",

                "",
                "",

                "",
                "",

                "",
                "",

                item[
                    "pred_x"
                ],

                item[
                    "pred_y"
                ],

                "",

                item[
                    "missing_frames"
                ],

                "",
                "motion_prediction",
                item["pred_x"],
                item["pred_y"],
                "",
            ]
        )

    # ========================================================
    # MINI MAP
    # ========================================================

    mini_map = create_mini_map(
        assignments,
        predicted,
        identity_manager
    )

    map_scale = min(
        1.0,
        height
        *
        0.62
        /
        MAP_HEIGHT
    )

    mini_display = cv2.resize(
        mini_map,
        (
            int(
                MAP_WIDTH
                *
                map_scale
            ),
            int(
                MAP_HEIGHT
                *
                map_scale
            )
        ),
        interpolation=cv2.INTER_AREA
    )

    mh, mw = (
        mini_display.shape[:2]
    )

    offset_x = (
        width
        -
        mw
        -
        20
    )

    offset_y = 20

    cv2.rectangle(
        display,
        (
            offset_x - 8,
            offset_y - 8
        ),
        (
            offset_x
            +
            mw
            +
            8,

            offset_y
            +
            mh
            +
            8
        ),
        (
            0,
            0,
            0
        ),
        -1
    )

    display[
        offset_y:
        offset_y + mh,

        offset_x:
        offset_x + mw
    ] = mini_display

    cv2.putText(
        display,
        (
            f"Detections: "
            f"{len(detections)} "
            f"Logical: "
            f"{len(assignments)}"
        ),
        (
            20,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    writer.write(
        display
    )

    processed += 1
    frame_index += 1

    if (
        processed
        %
        30
        ==
        0
    ):

        percent = (
            processed
            /
            frames_to_process
            *
            100
            if
            frames_to_process
            >
            0
            else
            0
        )

        print(
            f"\rProcessing: "
            f"{processed}/"
            f"{frames_to_process} "
            f"({percent:.1f}%)",
            end="",
            flush=True
        )


# ============================================================
# CLOSE
# ============================================================

cap.release()
writer.release()
csv_file.close()

if debug_tracks_file is not None:
    debug_tracks_file.close()

print()
print()
print(
    "========================================"
)

print(
    "PLAYER V2.1 COMPLETE"
)

print(
    "========================================"
)

print()
print(
    "Video:"
)
print(
    output_video_path
)

print()
print(
    "CSV:"
)
print(
    output_csv_path
)

if args.debug_tracks:
    print()
    print(
        "Raw track debug CSV:"
    )
    print(
        debug_tracks_csv_path
    )
