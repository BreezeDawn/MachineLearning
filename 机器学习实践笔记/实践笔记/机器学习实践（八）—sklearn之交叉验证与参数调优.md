## 一、交叉验证与参数调优

- #### 交叉验证(cross validation)

  - 交叉验证：将拿到的训练数据，分为训练集、验证集和测试集。
    - 训练集：训练集+验证集
    - 测试集：测试集

  <!-- more -->

  - 为什么需要交叉验证
    - 为了让被评估的模型更加稳健

- #### 参数调优

  - 超参数搜索-网格搜索(Grid Search)

    通常情况下，有很多参数是需要手动指定的（如k-近邻算法中的K值），这种叫超参数。但是手动过程繁杂，所以需要对模型预设几种超参数组合。每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

- #### 区分交叉验证和参数调优

  - 交叉验证
    - 使模型更稳健
  - 参数调优
    - 使模型准确性更高


## 二、模型选择、参数调优和交叉验证集成 API

sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)

- 介绍
  - 对估计器的指定参数值进行详尽搜索
- 参数介绍
  - estimator
    - 估计器对象
  - param_grid
    - 估计器参数(dict){“n_neighbors”:[1,3,5]}
  - cv
    - 指定几折交叉验证
- return
  - estimator
    - 新的估计器对象
- 使用新的估计器对象方法不变
  - fit：输入训练数据
  - score：准确率
- 新估计器对象的属性
  - best*score*:在交叉验证中验证的最好结果_
  - best*estimator*：最好的参数模型
  - cv*results*:每次交叉验证后的验证集准确率结果和训练集准确率结果

## 三、交叉验证与参数调优-案例：鸢尾花案例增加K值调优

- #### 完整代码

  ```python
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split,GridSearchCV
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
  
  # 指定算法及模型选择与调优——网格搜索和交叉验证
  estimator = KNeighborsClassifier()
  param_dict = {"n_neighbors": [1, 3, 5]}
  estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
  
  # 训练模型
  estimator.fit(x_train,y_train)
  
  # 模型评估
  # 方法一 比对真实值与预测值
  y_predict = estimator.predict(x_test)
  y_test == y_predict
  # 方法二 计算准确率
  estimator.score(x_test,y_test)
  
  # 然后进行评估查看最终选择的结果和交叉验证的结果
  print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
  print("最好的参数模型：\n", estimator.best_estimator_)
  print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)
  ```


## 四、交叉验证与参数调优-案例：预测facebook签到位置

- #### 目标

  - 将根据用户的位置，准确性和时间戳预测用户正在查看的业务。

- #### 数据集介绍

  ![](./img/FBlocation介绍.png)

  - 两个文件

    - train.csv
    - test.csv 

  - 文件字段

    - row_id：登记事件的ID

    - xy：坐标

    - accuracy：定位准确性 

    - time：时间戳

    - place_id：业务的ID，这是您预测的目标


  > 官网：<https://www.kaggle.com/navoshta/grid-knn/data>

- #### 步骤分析

  - 数据预处理
    - 缩小数据集范围
    - 时间特征提取
    - 将签到数少于n的位置删除
  - 数据集划分
  - 特征工程
    - 标准化
  - KNN算法
  - GSCV优化
  - 模型评估

- #### 完整代码

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#读取数据
facebook=pd.read_csv("./data/FBlocation/train.csv")
facebook.head()

# 数据预处理
# 1> 缩小数据集范围
facebook = facebook.query("x<1.5&x>1.25&y>2.25&y<2.5")
# 2> 时间特征提取
time_value = pd.to_datetime(facebook['time'],unit='s')
time_value = pd.DatetimeIndex(time_value)
facebook['day'] = time_value.day
facebook['hour'] = time_value.hour
facebook['weekday'] = time_value.weekday
# 3> 删除签到数少于n的位置
place_count = facebook.groupby(['place_id']).count()
place_count = place_count.query('row_id>3')
facebook = facebook[facebook['place_id'].isin(place_count.index)]

# 数据集划分
# 1> 拿取有用的特征数据
x=facebook[['x','y','accuracy','day','hour','weekday']]
# 2> 拿取目标值数据
y=facebook['place_id']
# 3> 数据集划分
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=8)

# 特征工程：标准化
# 1> 创建转换器
transfer = StandardScaler()
# 2> 计算并标准化训练集数据
x_train = transfer.fit_transform(x_train)
# 3> 计算并标准化测试集数据
x_test = transfer.transform(x_test)

# 模型训练及参数优化
# 1> 实例化一个K-近邻估计器
estimator = KNeighborsClassifier()
# 2> 运用网络搜索参数优化KNN算法
param_dict = {"n_neighbors":[3,5,7,9]}  # K-近邻中分别选取这几个 K 值，最终经过交叉验证会返回各个取值的结果和最好的结果
estimator = GridSearchCV(estimator,param_grid=param_dict,cv=5)  # 返回优化后的估计器
# 3> 传入训练集，进行机器学习
estimator.fit(x_train,y_train)

# 模型评估
# 方法一：比较真实值与预测值
y_predict=estimator.predict(x_test)
print("预测值为:\n",y_predict)
print("比较真实值与预测值结果为:\n",y_predict==y_test)
# 方法二：计算模型准确率
print("模型准确率为:\n",estimator.score(x_test,y_test))
print("在交叉验证中最的结果:\n",estimator.best_score_)
print("最好的参数模型:\n",estimator.best_estimator_)
print("每次交叉验证后的结果准确率为/n",estimator.cv_results_)
```
