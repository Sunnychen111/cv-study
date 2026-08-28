import numpy as np
from scipy.optimize import linear_sum_assignment

iou_cost = np.array([
    [0.10, 0.60],
    [0.55, 0.12]
], dtype=np.float32)

reid_cost = np.array([
    [0.80, 0.05],
    [0.06, 0.75]
], dtype=np.float32)

lambda_list = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]

for lambda_iou in lambda_list:

    # 1. 融合
    final_cost = lambda_iou * iou_cost + (1-lambda_iou) * reid_cost

    print(final_cost)

    # 2. Hungarian
    row_idx, col_idx = linear_sum_assignment(final_cost)

    print("lambda =", lambda_iou)

    print(final_cost)

    for track_idx, detection_idx in zip(row_idx, col_idx):
        print(
            track_idx,
            "->",
            detection_idx
        )

    print("----------------")