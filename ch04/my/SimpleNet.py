import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self) -> None:
        self.W: np.ndarray = np.random.randn(2, 3)  # 正規分布

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W)

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        z: np.ndarray = self.predict(x)
        y: np.ndarray = softmax(z)
        loss: float = cross_entropy_error(y, t)
        return loss


def main() -> None:
    pass
