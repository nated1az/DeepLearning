import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1 定义目标函数
# ===============================
def target_function(x):
    return np.sin(x) + 0.3 * x


# ===============================
# 2 生成数据集
# ===============================
def generate_dataset():
    np.random.seed(0)

    # 训练集（随机采样）
    x_train = np.random.uniform(-5, 5, 120).reshape(-1, 1)
    y_train = target_function(x_train)

    # 测试集（均匀采样）
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_test = target_function(x_test)

    return x_train, y_train, x_test, y_test


# ===============================
# 3 ReLU函数
# ===============================
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


# ===============================
# 4 初始化网络参数
# ===============================
def initialize_network():
    np.random.seed(1)

    W1 = np.random.randn(1, 64) * np.sqrt(2 / 1)
    b1 = np.zeros((1, 64))

    W2 = np.random.randn(64, 32) * np.sqrt(2 / 64)
    b2 = np.zeros((1, 32))

    W3 = np.random.randn(32, 1) * np.sqrt(2 / 32)
    b3 = np.zeros((1, 1))

    return W1, b1, W2, b2, W3, b3


# ===============================
# 5 前向传播
# ===============================
def forward(x, params):
    W1, b1, W2, b2, W3, b3 = params

    z1 = x @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    y_pred = a2 @ W3 + b3

    cache = (z1, a1, z2, a2)

    return y_pred, cache


# ===============================
# 6 训练函数
# ===============================
def train(x_train, y_train, epochs=20000, lr=0.005):

    W1, b1, W2, b2, W3, b3 = initialize_network()
    params = [W1, b1, W2, b2, W3, b3]

    loss_list = []

    for epoch in range(epochs):

        y_pred, cache = forward(x_train, params)
        z1, a1, z2, a2 = cache

        # MSE loss
        loss = np.mean((y_pred - y_train) ** 2)
        loss_list.append(loss)

        # =====================
        # 反向传播
        # =====================
        dL = 2 * (y_pred - y_train) / len(x_train)

        dW3 = a2.T @ dL
        db3 = np.sum(dL, axis=0, keepdims=True)

        da2 = dL @ W3.T
        dz2 = da2 * relu_grad(z2)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_grad(z1)

        dW1 = x_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # =====================
        # 参数更新
        # =====================
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

        params = [W1, b1, W2, b2, W3, b3]

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return params, loss_list


# ===============================
# 7 测试函数
# ===============================
def predict(x, params):
    y_pred, _ = forward(x, params)
    return y_pred


# ===============================
# 8 绘图并保存
# ===============================
def plot_results(x_train, y_train, x_test, y_pred, loss_list):

    # 拟合结果图
    plt.figure(figsize=(7,5))

    plt.scatter(x_train, y_train, label="Training Data", s=15)
    plt.plot(x_test, y_pred, color='red', label="Prediction")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Fitting with ReLU Network")

    plt.legend()

    plt.savefig("fitting_result.png", dpi=300, bbox_inches='tight')
    plt.show()


    # 损失曲线
    plt.figure(figsize=(7,5))

    plt.plot(loss_list)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")


    plt.show()


# ===============================
# 9 主程序
# ===============================
if __name__ == "__main__":

    x_train, y_train, x_test, y_test = generate_dataset()

    params, loss_list = train(x_train, y_train)

    y_pred = predict(x_test, params)

    plot_results(x_train, y_train, x_test, y_pred, loss_list)

    print("\n图像已保存为：")
    print("fitting_result.png")
