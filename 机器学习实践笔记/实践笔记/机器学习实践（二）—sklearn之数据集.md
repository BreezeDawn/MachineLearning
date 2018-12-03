## 一、可用数据集

+ Kaggle网址：https://www.kaggle.com/datasets
+ UCI数据集网址： http://archive.ics.uci.edu/ml/
+ scikit-learn网址：http://scikit-learn.org/stable/datasets/index.html

<!-- more -->

> 各数据集优点
>
> sk 数据量小，方便学习
>
> uci 数据真实，全面
>
> ka 竞赛平台，数据集真实

## 二、Scikit-learn

#### 1. 介绍

- Python语言的机器学习工具
- Scikit-learn包括许多知名的机器学习算法的实现
- Scikit-learn文档完善，容易上手，丰富的API
- 目前稳定版本0.19.1

#### 2. 安装

通过 pip 安装

```
pip3 install Scikit-learn==0.19.1
```

安装好之后可以通过以下命令查看是否安装成功

```python
import sklearn
```

> 注：安装scikit-learn需要Numpy, Scipy等库

#### 3. Scikit-learn 主要的API

1. 分类、聚类、回归
2. 特征工程
3. 模型选择、调优

## 三、SKlearn 数据集

#### 1. 数据集介绍

+ sklearn.datasets
  + load_*()
    + 获取小规模数据集，数据包含在datasets里
  + fetch_*(data_home=None)
    + 获取大规模数据集，需要从网络上下载。
    + 函数的第一个参数是data_home，表示数据集下载的目录,默认目录是根目录下的 scikit_learn_data文件夹： ~/scikit_learn_data/

#### 2. sklearn小数据集

+ 示例：
  + sklearn.datasets.load_iris()
    + 加载并返回鸢尾花数据集
  + sklearn.datasets.load_boston()
    + 加载并返回波士顿房价数据集

#### 3. sklearn大数据集

+ 示例：
  + sklearn.datasets.fetch_20newsgroups(data_home=None,subset=‘train’)
    + subset：'train'或者'test'，'all'，可选，选择要加载的数据集。
    + 训练集的“训练”，测试集的“测试”，两者的“全部”

#### 4. sklearn数据集返回值介绍

- load 和 fetch 返回的数据类型 datasets.base.Bunch (字典格式)
  - data：特征数据数组（特征值输入）
  - target：标签数组（目标输出）
  - feature_names：特征名称
  - target_names：标签名称
  - DESCR：数据描述
- Bunch 虽然是字典格式，但可以通过 '点' 的形式把属性点出来

+ 示例代码：

  ```python
  from sklearn.datasets import load_iris
  
  # 获取鸢尾花数据集
  iris = load_iris()
  
  print("鸢尾花数据集的返回值：\n", iris)
  
  print("鸢尾花的特征值:\n", iris["data"])
  
  print("鸢尾花的目标值：\n", iris.target)
  
  print("鸢尾花特征的名字：\n", iris.feature_names)
  
  print("鸢尾花目标值的名字：\n", iris.target_names)
  
  print("鸢尾花的描述：\n", iris.DESCR)
  ```


## 四、数据集划分

+ 机器学习一般的数据集会划分为两个部分：
  + 训练数据：用于训练、构建模型
  + 测试数据：在模型检验时使用，用于 评估模型是否有效

+ 划分比例：
  + 训练集：70~80%
  + 测试集：20~30%

+ 数据集划分 api
  + sklearn.model_selection.train_test_split( x, y, test_size, random_state )
    - x 数据集的特征值
    - y 数据集的标签值
    - test_size 测试集的大小，一般为float，默认 25%
    - random_state 随机数种子,不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
  + return 
    + 测试集特征
    + 训练集特征
    + 训练标签
    + 测试标签 (默认随机取)

+ 示例代码：

  ```python
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  
  # 获取鸢尾花数据集
  iris = load_iris()
  
  # 默认测试集占比 25%
  # 第一次划分，随机种子 22
  x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
  
  # 第二次划分，随机种子 6
  x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
  
  # 第三次划分，随机种子 6
  x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
  
  # 比较第一次和第二次划分，当随机种子设置不同时，划分结果不同
  print(x_train == x_train1)
  
  # 比较第二次和第三次划分，当随机种子设置相时，划分结果相同
  print(x_train1 == x_train2)
  
  ```





































