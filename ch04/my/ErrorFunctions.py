import numpy as np


def sum_squared_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:  # y: データ, t: 基準
    return 0.5 * np.sum((y - t) ** 2)


"""
教師データがont-hot表現の場合
"""


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        t.reshape(1, t.size)
        y.reshape(1, y.size)
        # y = [0.1,0.2,0.7]とかだったときy=[[0.1,0.2,0.7]]に変える処理
        # yの次元が2以上だった場合とそろえるため

    batchSize: int = y.shape[0]  # y=[[0.1,0.2,0.7]]なら、batchSize=1
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batchSize
