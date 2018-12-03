## 一、线性回归应用场景

- 房价预测
- 销售额度预测
- 金融：贷款额度预测、利用线性回归以及系数分析因子

<!-- more -->

## 二、线性回归的原理

+ #### 什么是回归

  在机器学习中，回归就是拟合的意思，我们需要找出一个模型来拟合(回归)数据。

+ #### 什么是线性回归

  + 线性回归是：利用回归方程(函数)，对特征值和目标值之间关系进行建模的一种分析方式。
  + 特征值和目标值可以是一个或多个，特征值和目标值可以看作函数意义上的自变量和因变量。

+ #### 特点

  + 只有一个自变量的情况称为单变量回归。
  + 多于一个自变量的情况称为多元回归。

+ #### 通用公式

  $$
  h(\theta) = \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... + b = \theta^Tx + b
  $$

  + 其中：
    + $\theta = (\theta_1,\theta_2,...,\theta_n,b)^T$
    + $x = (x_1,x_2,...,x_n,1)^T$

+ #### 线性回归的特征与目标的关系

  线性回归当中线性模型有两种，一种是线性关系，另一种是非线性关系。

## 三、损失函数

假设真实值为 $y$ ，我们的预测值为 $h(\theta)$ ，真实结果与我们预测的结果之间可能存在一定的误差。

既然存在这个误差，那我们就可以将这个误差给衡量出来。

+ #### 损失函数

  $$
  J(\theta) = (h_\theta(x_1)-y_1)^2 +(h_\theta(x_2)-y_2)^2 + ... +(h_\theta(x_3)-y_3)^2 = \sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
  $$

  - $y_i$ 为第 i 条训练数据的真实值。
  - $h(x_i)$ 为第 i 条训练数据的预测函数产生的预测值，且 $h(x_i)$ 是关于参数 $\theta$ 的函数。
  - 又称最小二乘法

## 四、优化算法

如何去减少损失函数的损失使我们预测的更加准确呢？

我们一直说机器学习有自动学习的能力，在线性回归这里更是能够体现。

这里可以通过一些优化方法去优化（本质是求导）回归的总损失！

+ #### 正规方程

  $$
  \theta = (X^TX)^{-1}X^Ty
  $$

  > 理解：X 为特征值矩阵，y 为目标值矩阵。直接求到最好的结果。
  >
  > 缺点：当特征过多过复杂时，求解速度太慢并且得不到结果。

+ #### 梯度下降(Gradient Descent)

  $$
  \theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta)
  $$

  > 理解：α 为学习速率，需要手动指定（超参数），$\frac{\partial}{\partial\theta_j} J(\theta)$ 整体表示下降方向
  >
  > 沿着这个函数下降的方向，最后就能找到山谷的最低点，然后更新 $\theta$ 值
  >
  > 使用：面对训练数据规模十分庞大的任务 ，能够找到较好的结果

  + 梯度下降步骤
    1. 随机初始化一个点 
    2. 自主学习
    3. 达到最小，终止学习

+ #### 有了梯度下降这样一个优化算法，回归就有了"自动学习"的能力

![](\img\线性回归优化动态图.gif)

## 五、sklearn - 线性回归 - API

sklearn提供给我们两种实现的API， 可以根据选择使用。

+ #### 正规方程 API

  sklearn.linear_model.LinearRegression(fit_intercept=True)

  - 通过正规方程优化
  - fit_intercept
    - 是否计算偏置
  - LinearRegression.coef_
    - 回归系数
  - LinearRegression.intercept_
    - 偏置

+ #### 梯度下降 API

  sklearn.linear_model.SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate ='invscaling', eta0=0.01)

  - SGDRegressor类实现了随机梯度下降学习，它支持不同的**loss函数和正则化惩罚项**来拟合线性回归模型。
  - loss
    - 损失类型
    - loss=”squared_loss”: 普通最小二乘法
  - fit_intercept
    - 是否计算偏置
  - learning_rate 
    - 学习率填充
    - 'constant'
      - $\eta = \eta_0$
    - 'optimal'
      - $\eta = \frac{1.0}{\alpha * (t + t_0)} $ [default]
    - 'invscaling'
      - $\eta = \frac{\eta_0}{pow(t, power_t)}$
      - power_t=0.25
        - 存在父类当中
    - 对于一个常数值的学习率来说，可以使用learning_rate=’constant’ ，并使用 $\eta_0$来指定学习率。
  - SGDRegressor.coef_
    - 回归系数
  - SGDRegressor.intercept_
    - 偏置

+ #### 回归性能评估 API

  + 均方误差(Mean Squared Error)评价机制：

  $$
  MSE = \frac{1}{m}\sum_{i=1}^{m}(y^i-y^{real})^2
  $$

  > 注：$y^i$ 为预测值，$y^{real}$ 为真实值

  + sklearn.metrics.mean_squared_error(y_true, y_pred)
    + 均方误差回归损失
    + y_true
      + 真实值
    + y_pred
      + 预测值
    + return
      + 浮点数结果

## 六、案例 - 线性回归 - 波士顿房价预测

+ #### 数据介绍

  ![](\img\房价数据集介绍.png)

![](\img\属性.png)

> 给定的这些特征，是专家们得出的影响房价的结果属性。我们此阶段不需要自己去探究特征是否有用，只需要使用这些特征。

+ #### 步骤分析

  回归当中的数据大小不一致，是否会导致结果影响较大。所以需要做标准化处理。

  - 数据分割与标准化处理
  - 回归预测
  - 线性回归的算法效果评估

+ #### 完整代码

  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression,SGDRegressor
  from sklearn.datasets import load_boston
  from sklearn.metrics import mean_squared_error
  
  # 获取数据
  boston = load_boston()
  
  # 划分数据集
  x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=8)
  
  # 特征工程，标准化
  # 1> 创建一个转换器
  transfer = StandardScaler()
  # 2> 数据标准化
  x_train = transfer.fit_transform(x_train)
  x_test = transfer.transform(x_test)
  
  # 方法一：正规方程求解
  # 模型训练
  # 1> 创建一个估计器
  estimator_1 = LinearRegression()
  # 2> 传入训练数据，进行机器学习
  estimator_1.fit(x_train,y_train)
  # 3> 打印梯度下降优化后的模型结果系数
  print(estimator_1.coef_)
  # 4> 打印梯度下降优化后的模型结果偏置
  print(estimator_1.intercept_)
  
  # 方法二：梯度下降求解
  # 模型训练
  # 1> 创建一个估计器，可以通过调参数，找到学习率效果更好的值
  estimator_2 = SGDRegressor(learning_rate='constant', eta0=0.001)
  # 2> 传入训练数据，进行机器学习
  estimator_2.fit(x_train,y_train)
  # 3> 打印梯度下降优化后的模型结果系数
  print(estimator_2.coef_)
  # 4> 打印梯度下降优化后的模型结果偏置
  print(estimator_2.intercept_)
  
  # 模型评估
  # 使用均方误差对正规方程模型评估
  y_predict = estimator_1.predict(x_test)
  error = mean_squared_error(y_test,y_predict)
  print('正规方程优化的均方误差为:\n',error)
  
  # 使用均方误差对梯度下降模型评估
  y_predict = estimator_2.predict(x_test)
  error = mean_squared_error(y_test,y_predict)
  print('梯度下降优化的均方误差为:\n',error)
  ```

## 七、正规方程与梯度下降比较

|       梯度下降       |            正规方程             |
| :------------------: | :-----------------------------: |
|    需要选择学习率    |             不需要              |
|     需要迭代求解     |          一次运算得出           |
| 特征数量较大可以使用 | 需要计算方程，时间复杂度高O(n3) |
|     适用大数据集     |          适用小数据集           |