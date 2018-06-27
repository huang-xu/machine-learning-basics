# 2.1 线性回归模型\(Linear Regression Model\)

## 2.1.1 模型

假设数据集表示为 $$D=\{x_i, t_i\}_{i=0}^{N}$$，  
其中$$t_i \in \mathcal{R}$$为目标变量，$$x_i \in \mathcal{R}^{m}$$为特征。  
参数 $$w_i \in \mathcal{R}, i \in (0, 1, ..., M)$$。  
定义


$$
y_i = \sum_{i=j}^{m}w_jx_{ij} + w_0
$$


为线性回归模型，其中$$y_i$$为输入是$$x_i$$时的输出。

用向量和矩阵表示，$$\mathbf{w} = [w_0, w_1, ..., w_m]^T$$，$$\mathbf{y} = [y_0, y_1, ... , y_n]^T$$  
$$\mathbf{X} = [x_{ij}], i\in (1,2,..., N), j\in (0,1,..., M)$$，其中$$x_{i0} = 1$$,则模型为


$$
\mathbf{y} = \mathbf{X} \mathbf{w}
$$


```py
import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def train(self, X, y):
        raise NotImplementedError('The train method must be implemented!')

    def test(self, X):
        try:
            self.w
        except:
            raise ValueError("Model must be trained before test!")
        return np.matmul(X, self.w)
```

## 2.1.2 示例数据



