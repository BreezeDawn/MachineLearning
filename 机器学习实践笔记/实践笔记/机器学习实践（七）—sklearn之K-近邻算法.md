## 一、K-近邻算法(KNN)原理

K Nearest Neighbor算法又叫KNN算法，这个算法是机器学习里面一个比较经典的算法， 总体来说KNN算法是相对比较容易理解的算法

<!-- more -->

- #### 定义

  如果一个样本在特征空间中的**k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别**，则该样本也属于这个类别。

  > 来源：KNN算法最早是由Cover和Hart提出的一种分类算法



- #### 距离公式

  两个样本的距离可以通过如下公式计算，又叫欧式距离
  $$
  distance = \sqrt{\sum_i^n (a_i-b_i)^2}
  $$

  > $A = (a_1,a_2,a_3,.....,a_n)$
  >
  > $B = (b_1,b_2,b_3,.....,b_n)$


## 二、简单实例-电影类型分析

#### 假设我们现在有几部电影



![电影类型分析](./img/电影类型分析.png)

#### 其中 $ ？$表示的电影不知道类别，如何去预测？我们可以利用K近邻算法的思想

![电影距离计算](./img/电影距离计算.png)

#### 问题

- 如果取的最近的电影数量不一样？会是什么结果？

  - k = 1 ，[爱情片]

  - k = 2 ，[爱情片，爱情片]

  - k = 3 ，[爱情片，爱情片，爱情片]

  - k = 4 ，[爱情片，爱情片，爱情片，动作片]

  - k = 6 ，[爱情片，爱情片，爱情片，动作片，动作片，动作片]

- 分析K-近邻算法需要做什么样的处理

  - k 是一个超参数，需要人为指定，好的 k 值更需要人的丰富经验
  - 当 k 取很大值时，受样本均衡影响较大
  - 当 k 取很小值时，受异常点影响较大

## 三、sklearn - KNN - API

sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')

- n_neighbors

  - 查询默认使用的邻居数
  - int ,可选（默认= 5）

- algorithm：

  - 可选用于计算最近邻居的算法：{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}

    - ‘ball_tree’将会使用 BallTree，
    - ‘kd_tree’将使用 KDTree。
    - ‘auto’将尝试根据传递给fit方法的值来决定最合适的算法。 
    - (不同实现方式影响效率)


## 四、KNN - 案例：鸢尾花种类预测

- #### 数据集介绍

  Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。关于数据集的具体介绍：

  ![](./img/iris%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D.png)

- #### 步骤分析

  - 获取数据集与分割数据集
  - 特征工程：标准化
  - 模型训练评估

- #### 完整代码

  ```python
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import KNeighborsClassifier
  
  # 加载数据
  iris = load_iris()
  
  # 划分数据集
  x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3,random_state=8)
  
  # 标准化
  transfer = StandardScaler()
  x_train = transfer.fit_transform(x_train)
  x_test = transfer.transform(x_test)
  
  # 训练模型
  estimator = KNeighborsClassifier()
  estimator.fit(x_train,y_train)
  
  # 模型评估
  # 方法一 比对真实值与预测值
  y_predict = estimator.predict(x_test)
  y_test == y_predict
  
  # 模型评估
  # 方法二 计算准确率
  estimator.score(x_test,y_test)
  ```


## 五、KNN优缺点

- 优点：
  - 简单，易于理解，易于实现，无需训练
- 缺点：
  - 懒惰算法，对测试样本分类时的计算量大，内存开销大
  - 必须指定K值，K值选择不当则分类精度不能保证
- 使用场景：小数据场景，几千～几万样本，具体场景具体业务具体分析