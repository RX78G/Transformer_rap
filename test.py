import math


def cross_entropy_loss(y_pred, y_actual):
    return -sum(a * math.log(p) for a, p in zip(y_actual, y_pred))


if __name__ == "__main__":
    # 予測された確率分布（例えば、ニューラルネットワークからの出力）
    y_pred = [0.1, 0.2, 0.7]

    # 実際のラベル（one-hotエンコーディングされた形）
    y_actual = [0, 0, 1]  # クラス3が正解

    loss = cross_entropy_loss(y_pred, y_actual)
    print(f"Cross-entropy loss: {loss:.3f}")
