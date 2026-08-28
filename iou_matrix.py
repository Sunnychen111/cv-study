# import numpy as np

# tracks = np.array([
#     [100, 100, 200, 200],   # Track 0
#     [300, 100, 400, 200]    # Track 1
# ], dtype=np.float32)

# detections = np.array([
#     [105, 105, 205, 205],   # Det 0，和 Track 0 很接近
#     [295, 100, 395, 200],   # Det 1，和 Track 1 很接近
#     [600, 100, 700, 200]    # Det 2，完全不相关
# ], dtype=np.float32)


# def compute_iou(box1,box2):
#     in_w=min(box1[2],box2[2])-max(box1[0],box2[0])
#     in_h=min(box1[3],box2[3])-max(box1[1],box2[1])

#     if(in_w<0 or in_h<0):
#         inner = 0
#     else:
#         inner =in_w*in_h

#     outer = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inner

#     iou= inner / outer
#     return iou

# def iou_matrix(tracks,detections):
#     matrix=np.zeros((len(tracks),len(detections)))
#     for i,track in enumerate(tracks):
#         for j,detection in enumerate(detections):
#             matrix[i,j]=compute_iou(track,detection)
#     return matrix

# print(iou_matrix(tracks,detections))

"""
匈牙利算法
import numpy as np
from scipy.optimize import linear_sum_assignment

cost_matrix = np.array([
    [0.10, 0.20],
    [0.11, 0.90]
], dtype=np.float32)

row_ind, col_ind = linear_sum_assignment(cost_matrix)

print("row:", row_ind)
print("col:", col_ind)

for track_idx, det_idx in zip(row_ind, col_ind):
    print(
        f"Track {track_idx} -> Det {det_idx}, "
        f"cost = {cost_matrix[track_idx, det_idx]}"
    )

"""

# 位置预测

def perdict_next_position(positions):
    i=1
    total=0
    while i<len(positions):
        total = total+positions[i]-positions[i-1] 
        i+=1
    avg = total/(len(positions)-1)
    return positions[len(positions)-1]+avg

print(perdict_next_position([100,105,110]))
print(perdict_next_position([200,197,194]))


      

























      





































































































































































































































































































































