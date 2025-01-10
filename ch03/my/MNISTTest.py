from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
import pickle
import my


def img_show(img) -> None:
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def loadTest() -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    # 試しに画像を表示
    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    img_show(img)


def getData():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


def initNetwork():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = my.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = my.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = my.softmax(a3)

    return y


def main() -> None:
    x, t = getData()
    network = initNetwork()
    acuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            acuracy_cnt += 1

    print("Accuracy: " + str(float(acuracy_cnt) / len(x)))


if __name__ == "__main__":
    main()
