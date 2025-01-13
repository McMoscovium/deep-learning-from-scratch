import numpy as np
import matplotlib.pylab as plt
from typing import Callable


def numericalDIff(f: Callable[[float], float], x: float) -> float:
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numericalGradient(
    f: Callable[[np.ndarray[float]], float], x: np.ndarray[float]
) -> np.ndarray[float]:
    if x.dtype != float:
        x = x.astype(float)  # dtype を float に変換

    h: float = 1e-4
    grad: np.ndarray[float] = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val: float = x[idx]  # x_iの値を保持
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1: float = f(x)
        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2: float = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numericalDescent(
    f: Callable[[np.ndarray], float], init_x: np.ndarray, lr=0.01, step_num: int = 100
) -> np.ndarray:
    """勾配降下法"""
    if init_x.dtype != float:
        init_x = init_x.astype(float)
    x: np.ndarray = init_x

    for i in range(step_num):
        grad: np.ndarray[float] = numericalGradient(f, x)
        x -= lr * grad

    return x


def f1(x: float) -> float:
    return 0.01 * x**2 + 0.1 * x


def f2(x: np.ndarray[float]) -> float:
    return np.sum(x**2)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
