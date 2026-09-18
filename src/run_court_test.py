from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import os


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEOS_DIR = PROJECT_ROOT / "Videos"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "court_benchmark"
)

TEST_VIDEO = (
    VIDEOS_DIR
    / "test.mp4"
)

TEST_VIDEO_BACKUP = (
    VIDEOS_DIR
    / "__test_backup.mp4"
)


# ============================================================
# COURT PIPELINE
#
# 注意：
# 前三個都是 intermediate。
#
# 真正 Court 成敗以最後：
#
# fit_birdeye_court_lines.py
#
# 的 refined_court_v2.json
# 和 refined_v2_original.jpg 為主。
# ============================================================

PIPELINE = [

    {
        "name": "AI coarse court",
        "script": "auto_court_stretch.py",
    },

    {
        "name": "CV bird-eye line detection",
        "script": "detect_birdeye_lines.py",
    },

    {
        "name": "Initial CV homography refinement",
        "script": "refine_court_homography.py",
    },

    {
        "name": "Final robust line fitting",
        "script": "fit_birdeye_court_lines.py",
    },
]


# ============================================================
# 每支影片測試前必須清掉
#
# 避免：
#
# Video A 的 refined_court.json
# 被 Video B 偷偷讀到
# ============================================================

CALIBRATION_FILES = [

    "auto_court.json",

    "court_detected_lines.json",

    "refined_court.json",

    "refined_court_v2.json",
]


OUTPUT_FILES = [

    "court_multiframe.jpg",

    "court_median_background.jpg",

    "court_line_detection.jpg",

    "refined_court_birdeye.jpg",

    "refined_court_original.jpg",

    "linefit_background.jpg",

    "linefit_debug.jpg",

    "refined_v2_birdeye.jpg",

    "refined_v2_original.jpg",
]


# ============================================================
# AUTO QUALITY THRESHOLDS
#
# 這只是幾何自動檢查。
#
# 最後仍然需要看：
#
# refined_v2_original.jpg
#
# 是否真的貼住白線。
# ============================================================

MIN_VERTICAL_LINES = 5
MIN_HORIZONTAL_LINES = 6

MIN_FINAL_INLIERS = 18

MAX_MEDIAN_INLIER_ERROR = 3.0


# ============================================================
# CLEAN OLD RUN
# ============================================================

def clean_previous_run():

    print()
    print("========================================")
    print("清除上一支影片的 Court 中間資料")
    print("========================================")


    for filename in CALIBRATION_FILES:

        path = (
            CALIBRATION_DIR
            / filename
        )


        if path.exists():

            print(
                "DELETE calibration:",
                filename
            )

            path.unlink()


    for filename in OUTPUT_FILES:

        path = (
            OUTPUTS_DIR
            / filename
        )


        if path.exists():

            print(
                "DELETE output:",
                filename
            )

            path.unlink()


# ============================================================
# RUN ONE PIPELINE STAGE
# ============================================================

def run_stage(
    stage_index,
    stage,
    log_lines
):

    script_path = (
        PROJECT_ROOT
        / "src"
        / stage["script"]
    )


    print()
    print()
    print(
        "========================================"
    )

    print(
        f"STAGE {stage_index + 1} / "
        f"{len(PIPELINE)}"
    )

    print(
        stage["name"]
    )

    print(
        stage["script"]
    )

    print(
        "========================================"
    )

    print()


    log_lines.append(
        ""
    )

    log_lines.append(
        "=" * 60
    )

    log_lines.append(
        f"STAGE {stage_index + 1}: "
        f"{stage['name']}"
    )

    log_lines.append(
        "=" * 60
    )


    env = os.environ.copy()

    

    # 避免 Windows 中文 console encoding 搞亂 log
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Headless wrapper:
    # server 沒有 GUI，所以停用 cv2.imshow / waitKey
    wrapper_code = f"""
import cv2
import runpy

cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: 0
cv2.destroyAllWindows = lambda *args, **kwargs: None
cv2.namedWindow = lambda *args, **kwargs: None
cv2.resizeWindow = lambda *args, **kwargs: None
cv2.moveWindow = lambda *args, **kwargs: None

runpy.run_path(
    {repr(str(script_path))},
    run_name="__main__"
)
"""

    process = subprocess.Popen(

        [
            sys.executable,
            "-c",
            wrapper_code
        ],

        cwd=str(
            PROJECT_ROOT
        ),

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True,

        encoding="utf-8",

        errors="replace",

        env=env
    )


    # 即時顯示 Terminal
    for line in process.stdout:

        print(
            line,
            end=""
        )

        log_lines.append(
            line.rstrip()
        )


    return_code = (
        process.wait()
    )


    if return_code != 0:

        raise RuntimeError(

            f"\nStage failed:\n"
            f"{stage['name']}\n"
            f"return code = "
            f"{return_code}"
        )


# ============================================================
# SAVE FILES
# ============================================================

def save_results(
    destination
):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )


    # Calibration
    for filename in CALIBRATION_FILES:

        source = (
            CALIBRATION_DIR
            / filename
        )


        if source.exists():

            shutil.copy2(

                source,

                destination
                / filename
            )


    # Images
    for filename in OUTPUT_FILES:

        source = (
            OUTPUTS_DIR
            / filename
        )


        if source.exists():

            shutil.copy2(

                source,

                destination
                / filename
            )


# ============================================================
# FINAL QUALITY CHECK
#
# 只讀 Final V2。
#
# 不用 auto_court_stretch 的圖判 PASS / FAIL。
# ============================================================

def evaluate_final_result():

    final_json = (
        CALIBRATION_DIR
        / "refined_court_v2.json"
    )


    if not final_json.exists():

        return {

            "status":
                "FAIL",

            "reason":
                "沒有產生 refined_court_v2.json"
        }


    with open(
        final_json,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    vertical_lines = data.get(
        "vertical_lines_fitted",
        []
    )


    horizontal_lines = data.get(
        "horizontal_lines_fitted",
        []
    )


    intersections = int(
        data.get(
            "intersection_count",
            0
        )
    )


    inliers = int(
        data.get(
            "intersection_inliers",
            0
        )
    )


    median_all = data.get(
        "median_all_error_px"
    )


    median_inlier = data.get(
        "median_inlier_error_px"
    )


    max_inlier = data.get(
        "max_inlier_error_px"
    )


    geometry_pass = (

        len(vertical_lines)
        >= MIN_VERTICAL_LINES

        and

        len(horizontal_lines)
        >= MIN_HORIZONTAL_LINES

        and

        inliers
        >= MIN_FINAL_INLIERS

        and

        median_inlier
        is not None

        and

        median_inlier
        <= MAX_MEDIAN_INLIER_ERROR
    )


    if geometry_pass:

        status = (
            "GEOMETRY_PASS"
        )

        reason = (
            "幾何數據通過；"
            "仍需人工看 refined_v2_original.jpg"
        )

    else:

        status = (
            "GEOMETRY_FAIL"
        )

        reason = (
            "最終 line-fitting 幾何數據未達門檻"
        )


    return {

        "status":
            status,

        "reason":
            reason,

        "vertical_lines":
            len(vertical_lines),

        "horizontal_lines":
            len(horizontal_lines),

        "intersections":
            intersections,

        "inliers":
            inliers,

        "median_all_px":
            median_all,

        "median_inlier_px":
            median_inlier,

        "max_inlier_px":
            max_inlier,
    }


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    destination,
    summary
):

    summary_json = (
        destination
        / "summary.json"
    )


    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            summary,

            f,

            indent=4,

            ensure_ascii=False
        )


    summary_txt = (
        destination
        / "summary.txt"
    )


    lines = [

        f"Video: "
        f"{summary.get('video')}",

        f"Status: "
        f"{summary.get('status')}",

        f"Reason: "
        f"{summary.get('reason')}",

        "",

        f"Vertical lines: "
        f"{summary.get('vertical_lines')}",

        f"Horizontal lines: "
        f"{summary.get('horizontal_lines')}",

        f"Intersections: "
        f"{summary.get('intersections')}",

        f"Inliers: "
        f"{summary.get('inliers')}",

        f"Median ALL: "
        f"{summary.get('median_all_px')} px",

        f"Median INLIER: "
        f"{summary.get('median_inlier_px')} px",

        f"Max INLIER: "
        f"{summary.get('max_inlier_px')} px",

        "",

        "IMPORTANT:",

        "Final visual judgment must use:",

        "refined_v2_original.jpg",
    ]


    summary_txt.write_text(

        "\n".join(lines),

        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) < 3:

    print()
    print("使用方式：")
    print()

    print(
        r"python src\run_court_test.py "
        r"Videos\影片.mp4 測試名稱"
    )

    print()

    print("例如：")

    print()

    print(
        r"python src\run_court_test.py "
        r"Videos\red_court.mp4 red_court"
    )

    sys.exit(1)


input_video = Path(
    sys.argv[1]
)


if not input_video.is_absolute():

    input_video = (
        PROJECT_ROOT
        / input_video
    )


input_video = (
    input_video.resolve()
)


run_name = (
    sys.argv[2]
)


if not input_video.exists():

    raise FileNotFoundError(

        f"找不到影片："
        f"{input_video}"
    )


destination = (

    RESULTS_ROOT
    / run_name
)


RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BACKUP CURRENT test.mp4
# ============================================================

had_original_test = (
    TEST_VIDEO.exists()
)


if TEST_VIDEO_BACKUP.exists():

    TEST_VIDEO_BACKUP.unlink()


if had_original_test:

    print()
    print(
        "備份目前 Videos/test.mp4"
    )


    shutil.copy2(

        TEST_VIDEO,

        TEST_VIDEO_BACKUP
    )


# ============================================================
# RUN TEST
# ============================================================

start_time = (
    time.time()
)


log_lines = []


pipeline_failed = False

failure_message = None


try:

    # --------------------------------------------------------
    # 清除上一部影片狀態
    # --------------------------------------------------------

    clean_previous_run()


    # --------------------------------------------------------
    # 先刪除舊 destination
    #
    # 避免之前結果混進來
    # --------------------------------------------------------

    if destination.exists():

        shutil.rmtree(
            destination
        )


    destination.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Set current test video
    # --------------------------------------------------------

    if (
        input_video
        != TEST_VIDEO.resolve()
    ):

        print()
        print(
            "暫時將測試影片複製成 "
            "Videos/test.mp4"
        )


        shutil.copy2(

            input_video,

            TEST_VIDEO
        )


    print()
    print(
        "========================================"
    )

    print(
        "COURT FULL PIPELINE TEST"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Video:"
    )

    print(
        input_video
    )

    print()

    print(
        "Run name:"
    )

    print(
        run_name
    )


    # --------------------------------------------------------
    # Run all stages
    # --------------------------------------------------------

    for stage_index, stage in enumerate(
        PIPELINE
    ):

        run_stage(

            stage_index,

            stage,

            log_lines
        )


    # --------------------------------------------------------
    # 最後才做正式幾何判定
    # --------------------------------------------------------

    summary = (
        evaluate_final_result()
    )


    summary[
        "video"
    ] = str(
        input_video
    )


    summary[
        "run_name"
    ] = run_name


    summary[
        "elapsed_seconds"
    ] = float(
        time.time()
        -
        start_time
    )


except Exception as error:

    pipeline_failed = True

    failure_message = str(
        error
    )


    summary = {

        "video":
            str(input_video),

        "run_name":
            run_name,

        "status":
            "PIPELINE_FAIL",

        "reason":
            failure_message,

        "elapsed_seconds":
            float(
                time.time()
                -
                start_time
            )
    }


finally:

    # --------------------------------------------------------
    # 不管成功失敗，
    # 都保存目前已產生的 debug
    # --------------------------------------------------------

    save_results(
        destination
    )


    # Pipeline log
    log_path = (
        destination
        / "pipeline.log"
    )


    log_path.write_text(

        "\n".join(
            log_lines
        ),

        encoding="utf-8"
    )


    save_summary(

        destination,

        summary
    )


    # --------------------------------------------------------
    # Restore original test.mp4
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )

    print(
        "恢復原本 Videos/test.mp4"
    )

    print(
        "========================================"
    )


    if had_original_test:

        if TEST_VIDEO_BACKUP.exists():

            shutil.copy2(

                TEST_VIDEO_BACKUP,

                TEST_VIDEO
            )


            TEST_VIDEO_BACKUP.unlink()


    else:

        if TEST_VIDEO.exists():

            TEST_VIDEO.unlink()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print(
    "========================================"
)

print(
    "FINAL COURT TEST RESULT"
)

print(
    "========================================"
)

print()

print(
    "Status:",
    summary.get(
        "status"
    )
)

print(
    "Reason:",
    summary.get(
        "reason"
    )
)


if (
    summary.get(
        "status"
    )
    != "PIPELINE_FAIL"
):

    print()

    print(
        "Vertical lines:",
        summary.get(
            "vertical_lines"
        )
    )

    print(
        "Horizontal lines:",
        summary.get(
            "horizontal_lines"
        )
    )

    print(
        "Intersections:",
        summary.get(
            "intersections"
        )
    )

    print(
        "Inliers:",
        summary.get(
            "inliers"
        )
    )

    print(
        "Median INLIER:",
        summary.get(
            "median_inlier_px"
        ),
        "px"
    )


print()
print(
    "完整結果："
)

print(
    destination
)

print()
print(
    "最終請人工檢查："
)

print(
    destination
    / "refined_v2_original.jpg"
)