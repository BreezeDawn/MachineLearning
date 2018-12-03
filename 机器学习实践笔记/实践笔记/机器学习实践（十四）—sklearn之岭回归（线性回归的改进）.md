带有 L2 正则化的线性回归就是岭回归。

岭回归，其实也是一种线性回归。

**只不过在算法建立回归方程时候，加上正则化的限制，从而达到解决过拟合的效果。**

加上正则化，也就是使权重满足划分正确结果的同时尽量的小

<!-- more -->

## 一、岭回归 - API

+ #### 岭回归 - API

  sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True,solver="auto", normalize=False)

  + 具有 L2 正则化的线性回归
  + alpha
    - 正则化力度，也叫 λ
    - λ 取值在0~1或者1~10
  + fit_intercept
    + 偏置，默认True
  + solver
    - 会根据数据自动选择优化方法
    - 如果数据集、特征都比较大，可以设置为 'sag' ，进行随机梯度下降优化
  + normalize
    - 数据是否进行标准化
    - normalize=True 时会进行标准化操作，我们就可以不使用 StandardScaler 进行标准化操作啦
  + Ridge.coef_
    - 回归权重
  + Ridge.intercept_
    - 回归偏置

  > All last four solvers support both dense and sparse data. However,
  > only 'sag' supports sparse input when `fit_intercept` is True.

+ #### SGDRegressor-API 实现岭回归

  SGDRegressor(penalty='l2', loss="squared_loss")

  + 使用梯度下降API实现岭回归
  + penalty
    + 乘法的意思，表示使用L2
  + loss 
    + 损失函数使用什么
    + squared_loss - 最小二乘

  + 推荐使用 Ridge，因为它实现了 SAG 优化

+ #### 岭回归 - 实现了交叉验证 - API

  sklearn.linear_model.RidgeCV(_BaseRidgeCV, RegressorMixin)

  - 具有 L2 正则化的线性回归，可以进行交叉验证
  - coef_
    - 回归系数

## 二、案例 - 岭回归 - 波士顿房价预测

+ #### 在之前案例基础上使用岭回归

+ #### 完整代码

  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import Ridge
  from sklearn.datasets import load_boston
  from sklearn.metrics import mean_squared_error
  
  # 获取数据
  boston = load_boston()
  
  # 划分数据集
  x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=6)
  
  # 特征工程：标准化
  # 1）实例化一个转换器类
  transfer = StandardScaler()
  # 2）调用fit_transform
  x_train = transfer.fit_transform(x_train)
  x_test = transfer.transform(x_test)
  
  # 岭回归的预估器流程
  estimator = Ridge()
  estimator.fit(x_train, y_train)
  y_predict = estimator.predict(x_test)
  print("岭回归求出模型参数的方法预测的房屋价格为：\n", y_predict)
  
  # 打印模型相关属性
  print("岭回归求出的回归系数为：\n", estimator.coef_)
  print("岭回归求出的偏置为：\n", estimator.intercept_)
  
  # 模型评估——均方误差
  error = mean_squared_error(y_test, y_predict)
  print("岭回归的均方误差为：\n", error)
  ```


