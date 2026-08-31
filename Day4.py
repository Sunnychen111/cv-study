import numpy as np
from scipy.optimize import linear_sum_assignment

x0 = np.array([
    [130.0],
    [20.0]
])

x1 = np.array([
    [250.0],
    [-20.0]
])

P0 = np.eye(2)
P1 = np.eye(2)


F = np.array([
    [1.0, 1.0],
    [0.0, 1.0]
])

Q = np.array([
    [0.1, 0.0],
    [0.0, 0.1]
])

H = np.array([
    [1.0, 0.0]
])

R = np.array([
    [4.0]
])

def predict(x,P,F,Q):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred,P_pred

def update(x,P,z):
    y =  z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    x_new = x + K @ y
    I = np.eye(len(x))
    P_new = (I - K @ H) @ P
    
    return x_new,P_new

def state_to_bbox(cx,P):
    x_pred,p_pred = predict(cx,P,F,Q)
    x=int(x_pred[0])
    bbox = [x-50,100,x+50,200]
    return bbox

def compute_iou(box1,box2):
    in_w=min(box1[2],box2[2])-max(box1[0],box2[0])
    in_h=min(box1[3],box2[3])-max(box1[1],box2[1])
    
    if(in_w<0 or in_h<0):
        inner = 0
    else:
        inner =in_w*in_h
    
    outer = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inner
    
    iou= inner / outer
    return iou

def iou_matrix(tracks,detections):
    matrix = np.zeros(
        [
            len(tracks),
            len(detections)
         ]
    )
    for i,track in enumerate(tracks):
        for j,detection in enumerate(detections):
            matrix[i,j] = compute_iou(track,detection)

    return matrix

# print("bbox0",state_to_bbox(x0,P0))
# print("bbox1",state_to_bbox(x1,P1))

tracks = ([state_to_bbox(x0,P0),state_to_bbox(x1,P1)])
detections = np.array([
    [170, 100, 270, 200],   # Detection 0
    [110, 100, 210, 200]    # Detection 1
], dtype=np.float32)

print(iou_matrix(tracks,detections))
iou_cost = 1-iou_matrix(tracks,detections)
print(iou_cost)

track_features = np.array([
    [0.90, 0.10],     # Track 0
    [0.10, 0.90]      # Track 1
], dtype=np.float32)
detection_features = np.array([
    [0.85, 0.15],     # Detection 0
    [0.15, 0.85]      # Detection 1
], dtype=np.float32)

def cosine_similarity(track_features,detection_features):
    similarity = np.zeros(
        [
            len(track_features),
            len(detection_features)
        ]
    )
    for i,track in enumerate(track_features):
        for j,detection in enumerate(detection_features):
            up = np.dot(track,detection)
            norm1 = np.linalg.norm(track)
            norm2 = np.linalg.norm(detection)
            similarity[i,j]=up/(norm1*norm2)
    return similarity

def bbox_center(box):
    cx = (box[0]+box[2]) / 2.0
    cy = (box[1]+box[3]) / 2.0
    return cx,cy



reid_cost = 1- cosine_similarity(track_features,detection_features)

lambda_iou = 0.4
final_cost = lambda_iou*iou_cost + (1-lambda_iou) *reid_cost
#匈牙利算法
row_idx,col_idx = linear_sum_assignment(final_cost)

for track_idx,detection_idx in zip(row_idx,col_idx):
    cx,cy=bbox_center(detections[detection_idx])
    print (
        track_idx,
        "->",
        detection_idx,
        "bbox_center:", cx,cy
    )
    if(track_idx==0):
        x_pred,P_pred = predict(x0,P0,F,Q)
        x0,P0 = update(x_pred,P_pred,cx)
        print(x0,P0)
    else:
        x_pred,P_pred = predict(x1,P1,F,Q)
        x1,P1 = update(x_pred,P_pred,cx)
        print(x1,P1)
