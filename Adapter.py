import numpy as np

"""
YOLO/RFDETR 与boxmot的适配

def build_boxmot_input(boxes, scores, class_ids):
    return np.column_stack((boxes,scores,class_ids)).astype(np.float32)

def rfdetr_to_boxmot(result):
    boxes = result.xyxy
    scores = result.confidence
    class_ids = result.class_id

    return build_boxmot_input(boxes,scores,class_ids)

def yolo_to_boxmot(result):
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    class_ids = result.boxes.cls

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    class_ids = class_ids.detach().cpu().numpy()

    return build_boxmot_input(boxes,scores,class_ids)

# boxes = np.array([
#     [100, 200, 300, 400],
#     [250, 120, 330, 240],
#     [500, 300, 650, 500]
# ], dtype=np.float32)

# scores = np.array([
#     0.91,
#     0.85,
#     0.73
# ], dtype=np.float32)

# class_ids = np.array([
#     0,
#     2,
#     1
# ])

boxes = np.empty((0, 4))
scores = np.empty((0,))
class_ids = np.empty((0,))

output=build_boxmot_input(boxes,scores,class_ids)
print(output)
print(output.shape)
print(output.dtype)

"""

# tracker数据分析
tracks = np.array([
    [100, 200, 300, 400, 1, 0.91, 0, 0],
    [250, 120, 330, 240, 5, 0.85, 2, 1],
    [500, 300, 650, 500, 8, 0.73, 1, 2]
], dtype=np.float32)

boxes = tracks[:,0:4]
track_ids = tracks[:,4]
scores = tracks[:,5]
class_ids = tracks[:,6]
mask = (scores>=0.8) & (class_ids==0)
boat_tracks = tracks[mask]

# print(boxes)
# print(track_ids)
# print(scores)
# print(class_ids)
# print(boat_tracks)

target_mask = (track_ids==1)
target_tracks=tracks[target_mask]
print(target_tracks)
x1=target_tracks[:,0]
x2=target_tracks[:,2]
y1=target_tracks[:,1]
y2=target_tracks[:,3]

#中心点位置
cx=(x2+x1)/2.0
cy=(y2+y1)/2.0
print(cx,cy)





