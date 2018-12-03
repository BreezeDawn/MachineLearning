## 一、sklearn转换器

+ 想一下之前做的特征工程的步骤？

  - 1 实例化 (实例化的是一个转换器类(Transformer))
  - 2 调用fit_transform(对于文档建立分类词频矩阵，不能同时调用)


<!-- more -->

+ 我们把特征工程的接口称之为转换器，其中转换器调用有这么几种形式
  + fit_transform
  + fit
  + transform

  **这几个方法之间的区别是什么呢？我们看以下代码就清楚了**

+ #### 示例代码

  ```python
  from sklearn.preprocessing import StandardScaler
  
  # 创建一个标准差转换器
  transfer = StandardScaler()
  a = [[1,2,3],[4,5,6]]
  
  # 进行计算均值和标准差，并进行转换，计算均值和标准差的结果会保存在transfer对象中，之后用到均值或标准差都会从对象中直接提取，如果重新计算会重新保存。
  transfer.fit_transform(a)
  
  # 进行均值和标准差的计算，保存在transfer对象中，
  transfer.fit(a)
  
  # 进行转换
  transfer.transform(a)
  ```

## 二、sklearn估计器

在sklearn中，估计器(estimator)是机器学习算法的API，是进行机器学习的面向对象，它的内部能够像转换器那样自动地保存一些运算结果。

+ #### 列举一些估计器
  + 1 用于分类的估计器：
    - sklearn.neighbors k-近邻算法
    - sklearn.naive_bayes 贝叶斯
    - sklearn.linear_model.LogisticRegression 逻辑回归
    - sklearn.tree 决策树与随机森林
  + 2 用于回归的估计器：
    - sklearn.linear_model.LinearRegression 线性回归
    - sklearn.linear_model.Ridge 岭回归
  + 3 用于无监督学习的估计器
    - sklearn.cluster.KMeans 聚类

+ #### 估计器工作流程

  + 实例化一个估计器

    ```python
    estimator = LNeighborsClassifier()
    ```

  + 传入训练数据集，进行机器训练

    ```python
    estimator.fit(x_train,y_train)
    ```

  + 模型评估

    + 方法1. 比较真实值与预测值

      ```python
      y_predict = estimator.predict(x_test)
      y_predict == y_test
      ```

    + 方法2. 计算模型准确率

      ```python
      estimator.score(x_test,y_test)
      ```