# 最小二乘法(Ordinary Least Squares)
最小二乘是线性回归模型常用的参数求解算法。

模型的目标函数是

$$
L(w, D) = \sum_{i=0}^{N} (t_i - y_i)^2 = \sum_{i=0}^{N} (t_i - \sum_{j=1}^m w_j x_{ij})^2
$$

即模型预测值与实际值的误差平方和。

写成矩阵形式为

$$
L(\mathbf{w}, D) = (\mathbf{t} - \mathbf{X} \mathbf{w})^T(\mathbf{t} - \mathbf{X} \mathbf{w})
$$

任务是求使得$$L(\mathbf{w}, D)$$取最小值的 $$\mathbf{w}$$。

由于$$L(\mathbf{w}, D)$$是$$\mathbf{w}$$的二次函数，如果$$L(\mathbf{w}, D)$$存在极值，则极值在$$\frac{\partial L(\mathbf{w}, D)}{\partial w} = 0$$处取得，因此求解 $$\frac{\partial L(\mathbf{w}, D)}{\partial w} = 0$$:

$$
\frac{\partial L(\mathbf{w}, D)}{\partial \mathbf{w}} = - \mathbf{X}^T   (\mathbf{t} - \mathbf{X} \mathbf{w}) =0
$$

得

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T \mathbf{t}
$$

解中包含求逆运算 $$(\mathbf{X}^T\mathbf{X})^{-1} $$，因此当 $$\mathbf{X}^T\mathbf{X}$$ 不可逆时最小二乘法不适用。

## 实现最小二乘
下面是最小二乘的实现
```py
class LinearOLS(LinearRegression):
    def __init__(self):
        super().__init__()

    def train(self, X, t):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(t))
        return self
```
其中 $$X$$ 和 $$t$$ 可能不是合适的形式，因此我们在加上一个数据类型转换。
```py
class LinearOLS(LinearRegression):
    def __init__(self):
        super().__init__()

    def train(self, X, t):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(t) is not np.ndarray:
            t = np.array(t).reshape([-1, 1])
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(t))
        return self
```