import numpy as np
import cv2

from boxmot.trackers.bbox.hybridsort import HybridSort

tracker=HybridSort(
    reid_model=None,
    with_reid=False,
    with_longterm_reid=False,
    with_longterm_reid_correction=False
)
np.set_printoptions(
    suppress=True,
    precision=2
)


def build_boxmot_input(boxes, scores, class_ids):
    
    return np.column_stack(
        (boxes, scores, class_ids)
    ).astype(np.float32)


def rfdetr_to_boxmot(result):
    boxes = result.xyxy
    scores = result.confidence
    class_ids = result.class_id

    return build_boxmot_input(
        boxes,
        scores,
        class_ids
    )


def yolo_to_boxmot(result):
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    class_ids = result.boxes.cls

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    class_ids = class_ids.detach().cpu().numpy()

    return build_boxmot_input(
        boxes,
        scores,
        class_ids
    )


def fake_tracker_update(dets):  #这个代替BOXMOT的使用
    """
    dets:
        [N, 6]
        [x1, y1, x2, y2, conf, cls]

    return:
        [N, 8]
        [x1, y1, x2, y2, track_id, conf, cls, det_index]
    """

    if len(dets) == 0:
        return np.empty((0, 8), dtype=np.float32)

    n = len(dets)

    track_ids = np.arange(1, n + 1).reshape(-1, 1)
    det_indices = np.arange(n).reshape(-1, 1)

    tracks = np.concatenate(
        [
            dets[:, :4],
            track_ids,
            dets[:, 4:6],
            det_indices
        ],
        axis=1
    )

    return tracks.astype(np.float32)

# boxes = np.array([
#     [100, 100, 300, 300],
#     [400, 150, 550, 320]
# ], dtype=np.float32)

# scores = np.array([
#     0.91,
#     0.82
# ], dtype=np.float32)

# class_ids = np.array([
#     0,
#     2
# ], dtype=np.float32)

# dets = build_boxmot_input(boxes,scores,class_ids)
# tracks=tracker.update(dets)

# print(tracks)

# frame = np.zeros(
#     (720, 1280, 3),
#     dtype=np.uint8
# )

# for track in tracks:

#     x1 = int(track[0])
#     y1 = int(track[1])
#     x2 = int(track[2])
#     y2 = int(track[3])

#     track_id = int(track[4])
#     score = track[5]
#     class_id = int(track[6])

#     cv2.rectangle(
#         frame,
#         (x1,y1),
#         (x2,y2),
#         (0,255,0),
#         2
#     )
#     label = f"ID:{track_id} CLS:{class_id} {score:.2f}"
#     cv2.putText(
#         frame,
#         label,
#         (x1, y1 - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (0, 255, 0),
#         2
#     )
# cv2.imshow("Tracking Demo", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for frame_idx in range(20):
    frame= np.zeros(
        (720,1280,3),
        dtype=np.uint8
    )
    x1 = 100 + frame_idx * 5
    y1 = 100
    x2 = 300 + frame_idx * 5
    y2 = 300

    B_x1 = 700+frame_idx*3
    B_y1 = 100
    B_x2 = 900 +frame_idx*3
    B_y2 = 300
    if frame_idx<8 and frame_idx>=3: #设计几个漏检的帧
        dets=np.empty((0,6),dtype=np.float32)

    elif frame_idx %2==0:
        boxes = np.array([
            [x1, y1, x2, y2],
            [B_x1,B_y1,B_x2,B_y2]
        ], dtype=np.float32)
        scores = np.array([
                0.91,
                0.85
            ], dtype=np.float32)
        class_ids = np.array([
                0,
                0
            ], dtype=np.float32)
        dets = build_boxmot_input(
                boxes,
                scores,
                class_ids
            )
    
    else:
        boxes = np.array([
            [B_x1,B_y1,B_x2,B_y2],
            [x1, y1, x2, y2]
        ], dtype=np.float32)
        scores = np.array([
                0.85,
                0.91
            ], dtype=np.float32)
        class_ids = np.array([
                0,
                0
            ], dtype=np.float32)
        dets = build_boxmot_input(
                boxes,
                scores,
                class_ids
            )
   
    tracks = tracker.update(
        dets,
        frame
    )

    
    print(
        "frame:",
        frame_idx,
    )
    for track in tracks:
        print(
            "ID=",
            int(track[4]),
            " | ",
            "Box=",
            track[0:4],
            " | ",
            "Conf=",
            track[5]
        )