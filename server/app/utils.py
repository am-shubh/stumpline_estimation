import cv2
import numpy as np
import json

from .config import *
from fastapi import Response


def get_prediction(video_path: str) -> str:
    """
    TODO:
    1. Download or read video streams.
    2. For each frame or group of frame, run inference
    3. Store those prediction as json file.
    """

    # Hard coded as of now
    return "/detections_sample_video.json"


def load_annotation_file(annotation_file_path: str):
    try:
        with open(annotation_file_path, "r") as fp:
            annot_data = json.load(fp)

        return annot_data

    except Exception as e:
        return Response(e, status_code=500)


def get_stumpline_coords(detections, frame_height):
    """
    - Assumption 1: The stumps at bowler end and batsmen end can be easily grouped using frame height
    - Assumption 2: Stumps at bowler end shuld have only three stumps i.e. one group. This assumption is based on the camera setup.
    - Assumption 3: The distance/offset between two consecutive stumps should be less than 30 pixels. This can be set after some analysis or heuristics.
    - Assumption 4: The center offset between stumps group at both end should be in range of 150 pixels.
    """

    detections_grouped = {"batsmen_end": [], "bowler_end": []}

    # splitting the detections using frame height i.e Assumption 1
    # this gives multiple detections stumps at batsmen end and bowler ends
    for detection in detections:
        if detection["class_id"] == CLASS_ID:
            if detection["y"] >= frame_height // 2:
                detections_grouped["bowler_end"].append(
                    [
                        detection["class_id"],
                        detection["det_id"],
                        detection["confidence"],
                        detection["bbox"],
                        detection["x"],
                        detection["y"],
                        detection["width"],
                        detection["height"],
                        detection["center"],
                    ]
                )
            else:
                detections_grouped["batsmen_end"].append(
                    [
                        detection["class_id"],
                        detection["det_id"],
                        detection["confidence"],
                        detection["bbox"],
                        detection["x"],
                        detection["y"],
                        detection["width"],
                        detection["height"],
                        detection["center"],
                    ]
                )

    # Stumps at bowler end should have only one group i.e. Assumption 2
    stumps_bowler = sorted(detections_grouped["bowler_end"], key=lambda item: item[4])
    stumps_bowler_center_x = (stumps_bowler[0][3][0] + stumps_bowler[-1][3][2]) // 2

    # Stumps at batsmen end or in the upper part of FOV, may have multiple stump positions
    # So we will group these stumps first and then find the stump which is closer to batsmen end
    stumps_batsmen = sorted(detections_grouped["batsmen_end"], key=lambda item: item[4])
    group = []

    # Sometime there may not be any prediction at batsmen end due to occlusion
    i = 0
    while i < len(stumps_batsmen):
        # Consecutive stumps should not have offset of more than 30 pixels i.e. Assumption 3
        if (
            i + 1 < len(stumps_batsmen)
            and abs(stumps_batsmen[i][4] - stumps_batsmen[i + 1][4])
            <= CONSECUTIVE_OFFSET
        ):
            if (
                i + 2 < len(stumps_batsmen)
                and abs(stumps_batsmen[i + 1][4] - stumps_batsmen[i + 2][4])
                <= CONSECUTIVE_OFFSET
            ):
                group.append(
                    [
                        stumps_batsmen[i],
                        stumps_batsmen[i + 1],
                        stumps_batsmen[i + 2],
                    ]
                )
                i = i + 3
            else:
                group.append([stumps_batsmen[i], stumps_batsmen[i + 1]])
                i = i + 2
        else:
            group.append([stumps_batsmen[i]])
            i = i + 1

    min_distance = 999999
    if len(group):
        # finding the stump which is closer to stumps group at bowler end.
        for grp in group:
            # compare only if two stumps were found in one group
            if len(grp) >= 2:
                grp_center_x = (grp[0][3][0] + grp[-1][3][2]) // 2
                _distance = abs(grp_center_x - stumps_bowler_center_x)

                # Assumption 4
                if _distance < min_distance and _distance <= STUMPS_OFFSET:
                    min_distance = _distance
                    stumps_batsmen = grp

    if min_distance != 999999:
        # Calculating the four key-points for particular frame
        top_left = (stumps_batsmen[0][3][0], stumps_batsmen[0][3][3])
        top_right = (stumps_batsmen[-1][3][2], stumps_batsmen[-1][3][3])
        bottom_left = (stumps_bowler[0][3][0], stumps_bowler[0][3][3])
        bottom_right = (stumps_bowler[-1][3][2], stumps_bowler[-1][3][3])

        return (top_left, top_right, bottom_right, bottom_left)

    else:
        return None


def visualize_stumpline(key_points, frame):
    pts = np.array(key_points, np.int32)

    overlay = frame.copy()

    cv2.fillPoly(overlay, [pts], color=OVERLAY_COLOR)

    img_new = cv2.addWeighted(
        overlay, TRANSPARENCY_FACTOR, frame, 1 - TRANSPARENCY_FACTOR, 0
    )

    return img_new


def get_average_keypoints(frames_keypoints):
    top_left_avg = (
        np.mean(np.array(frames_keypoints)[:, 0][:, 0]),
        np.mean(np.array(frames_keypoints)[:, 0][:, 1]),
    )
    top_right_avg = (
        np.mean(np.array(frames_keypoints)[:, 1][:, 0]),
        np.mean(np.array(frames_keypoints)[:, 1][:, 1]),
    )
    bottom_right_avg = (
        np.mean(np.array(frames_keypoints)[:, 2][:, 0]),
        np.mean(np.array(frames_keypoints)[:, 2][:, 1]),
    )
    bottom_left_avg = (
        np.mean(np.array(frames_keypoints)[:, 3][:, 0]),
        np.mean(np.array(frames_keypoints)[:, 3][:, 1]),
    )

    key_points_avg = (top_left_avg, top_right_avg, bottom_right_avg, bottom_left_avg)

    return key_points_avg
