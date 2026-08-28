import numpy as np

x = np.array([
    [100.0],
    [5.0]
])

P = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

F = np.array([
    [1.0, 1.0],
    [0.0, 1.0]
])

Q = np.array([
    [0.1, 0.0],
    [0.0, 0.1]
])


R = np.array([
    [4.0]
])

detections = [
    101.2,
    103.8,
    111.5,
    114.1,
    122.0,
    123.5,
    131.8,
    133.7,
    141.4,
    144.2
]


def perdict(x,P,F,Q):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred,P_pred

def update(x,P,z):
    H=np.array([
        [1.0,0.0]  #用于提取位置信息
    ])
    y =  z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x_new = x + K @ y
    I = np.eye(len(x))
    P_new = (I - K @ H) @ P

    return x_new,P_new
i=0
while i<len(detections):
    z = np.array([
        detections[i]
    ])
    x_pred,P_pred=perdict(x,P,F,Q)
    x,P=update(x_pred,P_pred,z)
    print(
        "Frame",
        i,
        " | Det:",
        detections[i],
        " | Pred:",
        x_pred,
        "| Update",
        x
    )
    i+=1



