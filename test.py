import numpy as np

def cross_entropy_loss(y_pred, y_actual):
    return -np.sum(y_actual * np.log(y_pred))

# 予測された確率分布（例えば、ニューラルネットワークからの出力）
y_pred = np.array([0.1, 0.2, 0.7])

# 実際のラベル（one-hotエンコーディングされた形）
y_actual = np.array([0, 0, 1]) # クラス3が正解

loss = cross_entropy_loss(y_pred, y_actual)
print(f"Cross-entropy loss: {loss:.3f}")
