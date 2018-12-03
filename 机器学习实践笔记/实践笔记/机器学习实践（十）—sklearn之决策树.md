## 一、决策树分类概述

- #### 介绍

  决策树思想的来源非常朴素，程序设计中的条件分支结构就是if-else结构，最早的决策树就是利用这类结构分割数据的一种分类学习方法。

- #### 原理

  - 信息熵
  - 信息增益


<!-- more -->

## 二、信息熵

- #### 定义

  - $H$ 的专业术语称之为信息熵，单位为**比特**。

- #### 公式

  $$
  H(X) = \sum_{i=1}^{n}p(x_i)I(x_i) = - \sum_{i=1}^{n}P(x_i)log_2P(x_i)
  $$




  > + $I(x)$ 用来表示随机变量的信息，$p(x_i)$ 指是当 $x_i$ 发生时的概率。
  > + log可以以别的数为底，只不过值会不同罢了
  > + 熵只依赖X的分布，和X的取值没有关系，熵是用来度量不确定性，当熵越大，概率说X=xi的不确定性越大，反之越小，在机器学期中分类中说，**熵越大即这个类别的不确定性更大，反之越小**

+ #### 来自百度百科

  > - 信息是个很抽象的概念。人们常常说信息很多，或者信息较少，但却很难说清楚信息到底有多少。比如一本五十万字的中文书到底有多少[信息量](https://baike.baidu.com/item/%E4%BF%A1%E6%81%AF%E9%87%8F/420401)。
  >
  > - 直到1948年，香农提出了“信息熵”的概念，才解决了对信息的量化度量问题。信息熵这个词是C．E．香农从热力学中借用过来的。热力学中的热熵是表示分子状态混乱程度的物理量。**香农用信息熵的概念来描述信源的不确定度。**
  > - [信息论](https://baike.baidu.com/item/%E4%BF%A1%E6%81%AF%E8%AE%BA/302185)之父[克劳德·艾尔伍德·香农](https://baike.baidu.com/item/%E5%85%8B%E5%8A%B3%E5%BE%B7%C2%B7%E8%89%BE%E5%B0%94%E4%BC%8D%E5%BE%B7%C2%B7%E9%A6%99%E5%86%9C/10588593)第一次用数学语言阐明了概率与信息冗余度的关系。

+ #### 来自<数学之美>

  > + 系统中各种随机性的概率越均等，信息熵越大，反之越小。
  > + 从香农给出的数学公式上可以看出，**信息熵其实是一个随机变量信息量的数学期望。**
  > + 查看更多：https://blog.csdn.net/saltriver/article/details/53056816

+ #### 示例助解

  举一个的例子：对游戏活跃用户进行分层，分为高活跃、中活跃、低活跃，游戏A按照这个方式划分，用户比例分别为20%，30%，50%。游戏B按照这种方式划分，用户比例分别为5%，5%，90%。那么游戏A对于这种划分方式的熵为：
  $$
  H(A) = -(0.2*log_{2}0.2+0.3*log_{2}0.3+0.5*log_{2}0.5) = 1.485
  $$
  同理游戏B对于这种划分方式的熵为：
  $$
  H(B) = -(0.05*log_{2}0.05+0.05*log_{2}0.05+0.9*log_{2}0.9) = 0.569
  $$
  游戏A的熵比游戏B的熵大，所以游戏A的不确定性比游戏B高。用简单通俗的话来讲，游戏B要不就在上升期，要不就在衰退期，它的未来已经很确定了，所以熵低。而游戏A的未来有更多的不确定性，它的熵更高。

## 三、 信息增益 - 决策树的划分依据之一

- #### 定义

  **特征A 对 训练数据集D 的 信息增益 g(D,A)** ,定义为 **集合D 的 信息熵H(D)** 与 **特征A条件下 D 的 信息条件熵H(D|A)** **的差。**

- #### 公式

  $$
  g(D,A) = H(D) - H(D|A)
  $$

- #### 公式的详细解释：

  - 信息熵的计算：
    $$
    H(D) = - \sum^{K}_{k=1}\frac{|C_k|}{|D|}log_{2}\frac{|C_k|}{|D|}
    $$

    - 条件熵的计算：

  $$
  H(D|A)=\sum_{i=1}^{n}\frac{|D_i|}{|D|}H(D_i)=- \sum_{i=1}^{n}\frac{|D_i|}{|D|}\sum^{K}_{k=1}\frac{|D_{ik}|}{|D_i|}log\frac{|D_{ik}|}{|D_i|}
  $$




  > 注：
  >
  > + $C_k$ 表示属于某个类别的样本数
  > + 信息增益表示得知特征X的信息而息的不确定性减少的程度使得类Y的信息熵减少的程度
  >
  > >
  > > 这句话应该有点毛病，太绕了，没有理解透，马上补，然后再回来修改

+ #### 示例助解

  假设有下表样本：

  第一列为QQ，第二列为性别，第三列为活跃度，最后一列用户是否流失。

  我们要解决一个问题：性别和活跃度两个特征，哪个对用户流失影响更大？我们通过计算信息熵可以解决这个问题。

  |  QQ  | gender | active_info | is_lost |
  | :--: | :----: | :---------: | :-----: |
  |  1   |   男   |     高      |    0    |
  |  2   |   女   |     中      |    0    |
  |  3   |   男   |     低      |    1    |
  |  4   |   女   |     高      |    0    |
  |  5   |   男   |     高      |    0    |
  |  6   |   男   |     中      |    0    |
  |  7   |   男   |     中      |    1    |
  |  8   |   女   |     中      |    0    |
  |  9   |   女   |     低      |    1    |
  |  10  |   女   |     中      |    0    |
  |  11  |   女   |     高      |    0    |
  |  12  |   男   |     低      |    1    |
  |  13  |   女   |     低      |    1    |
  |  14  |   男   |     高      |    0    |
  |  15  |   男   |     高      |    0    |

  按照分组统计，我们可以得到如下信息：

  |      | 已流失(人) | 未流失(人) | 汇总(人) |
  | :--: | :--------: | :--------: | :------: |
  | 整体 |     5      |     10     |    15    |
  |  男  |     3      |     5      |    8     |
  |  女  |     2      |     5      |    7     |
  |  高  |     0      |     6      |    6     |
  |  中  |     1      |     4      |    5     |
  |  低  |     4      |     0      |    4     |

  那么可得到三个熵：

  整体熵：
  $$
  H(S) = -\frac{5}{15}log_{2}(\frac{5}{15})-\frac{10}{15}log_{2}(\frac{10}{15}) = 0.9182
  $$
  性别熵：
  $$
  H(g_1) = -\frac{3}{8}log_{2}(\frac{3}{8}) - \frac{5}{8}log_{2}(\frac{5}{8}) = 0.9543
  $$

  $$
  H(g_1) = -\frac{2}{7}log_{2}(\frac{2}{7}) - \frac{5}{7}log_{2}(\frac{5}{7}) = 0.8631
  $$
  性别信息增益：
  $$
  g(S,g) = H(S)-\frac{8}{15}H(g_1)-\frac{7}{15}H(g_2)=0.0064
  $$
  同理计算活跃度熵：
  $$
  H(a_1)=0
  $$

  $$
  H(a_2)=0.7219
  $$

  $$
  H(a_3)=0.0
  $$

  活跃度信息增益：
  $$
  g(S,a) = H(S)-\frac{6}{15}H(a_1)-\frac{5}{15}H(a_2)-\frac{4}{15}H(a_3)=0.6776活跃度的信息增益比性别的信息增益大，也就是说，活跃度对用户流失的影响比性别大。
  $$
  活跃度的信息增益比性别的信息增益大，也就是说，活跃度对用户流失的影响比性别大。

  在做特征选择或者数据分析的时候，我们应该重点考察活跃度这个指标。

  > 参考：https://blog.csdn.net/guomutian911/article/details/78599450

### 三、决策树的其它划分依据

决策树的原理不止信息增益这一种，还有其他方法。

- ID3
  - 信息增益 最大的准则
- C4.5
  - 信息增益比 最大的准则
- CART
  - 分类树: 基尼系数 最小的准则 在sklearn中可以选择划分的默认原则
  - 优势：划分更加细致（从后面例子的树显示来理解）

## 四、决策树 API

class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)

- 决策树分类器
- criterion
  - 默认是’gini’系数
  - 也可以选择信息增益的熵’entropy’
- max_depth
  - 树的深度大小
  - 不指定树的深度很容易出现过拟合
- random_state
  - 随机数种子

> gini - 基尼系数

## 五、决策树 - 案例：鸢尾花分类案例

- 完整代码

  ```python
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.tree import DecisionTreeClassifier
  
  # 获取数据集
  iris=load_iris()
  
  # 分割数据集
  x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=8)
  
  # 特征工程：标准化
  transfer=StandardScaler()
  x_train=transfer.fit_transform(x_train)
  x_test=transfer.transform(x_test)
  
  # 模型训练
  # 1> 实例化一个估计器
  estimator=DecisionTreeClassifier(criterion='entropy',max_depth=3)
  # 2> 传入训练数据集，进行机器学习
  estimator.fit(x_train,y_train)
  
  # 模型评估
  # 方法1，比较真实值与预测值
  y_predict=estimator.predict(x_test)
  print("预测值为:\n",y_predict)
  print("比较真实值与预测值结果为:\n",y_predict==y_test)
  # 方法2, 计算模型准确率
  print("模型准确率为:\n",estimator.score(x_test,y_test))
  ```

## 六、决策树 - 案例：泰坦尼克号乘客生存预测

- #### 泰坦尼克号数据

  - 在泰坦尼克号和titanic2数据帧描述泰坦尼克号上的个别乘客的生存状态。这里使用的数据集是由各种研究人员开始的。其中包括许多研究人员创建的旅客名单，由Michael A. Findlay编辑。我们提取的数据集中的特征是票的类别，存活，乘坐班，年龄，登陆，home.dest，房间，票，船和性别。
  - 乘坐班是指乘客班（1，2，3），是社会经济阶层的代表。
  - 其中age数据存在缺失。

  > 数据：<http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt>

- #### 步骤分析

  - 数据预处理，填充缺失值
  - 提取特征值，目标值
  - 特征工程，字典特征提取
  - 数据集划分
  - 模型训练
  - 模型评估

- #### 完整代码

  ```python
  import pandas as pd
  from sklearn.feature_extraction import DictVectorizer
  from sklearn.model_selection import train_test_split
  from sklearn.tree import DecisionTreeClassifier
  
  # 获取数据
  tanic=pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
  tanic.head()
  
  # 数据预处理，填充缺失值
  tanic['age'].fillna(tanic['age'].mean(),inplace=True)
  
  # 提取特征值，目标值
  x=tanic[['pclass','age','sex']]
  y=tanic['survived']
  
  # 特征工程，字典特征提取
  transfer=DictVectorizer(sparse=False)
  x=transfer.fit_transform(x.to_dict(orient="records"))
  
  # 数据集划分
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
  
  # 模型训练
  # 1> 实例化一个转换器
  estimator=DecisionTreeClassifier()
  # 2> 进行机器学习
  estimator.fit(x_train,y_train)
  
  # 模型评估
  # 方法1，比较真实值与预测值
  y_predict=estimator.predict(x_test)
  print("预测值为:\n",y_predict)
  print("比较真实值与预测值结果为:\n",y_predict==y_test)
  # 方法2,计算模型准确率
  print("模型准确率为:\n",estimator.score(x_test,y_test))
  
  ```

## 七、决策树优缺点

+ 优点

  + 直观
  + 简单

+ #### 缺点

  + 容易过拟合