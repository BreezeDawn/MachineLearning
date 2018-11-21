## 一、特征预处理概述

+ #### 什么是特征预处理

  ```
  # scikit-learn的解释
  provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
  ```

  翻译过来：通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程

<!-- more -->

+ #### 数值型数据的无量纲化：
  + 归一化
  + 标准化

+ #### 为什么我们要进行归一化/标准化？

  [百面机器学习-为什么需要对数值类型的特征做归一化](http://xingtu.info/posts/6132/)

  如果存在特征的数值差别比较大的特征，那么分析出来的结果显然就会倾向于数值差别比较大的特征。

  如果存在特征的方差比较大的特征，那么分析出来的结果显然就会倾向于方差比较大的特征。

  我们需要用到一些方法进行无量纲化，使不同规格的数据转换到同一规格。

## 二、归一化

+ #### 定义

  通过对原始数据进行变换把数据映射到 [0,1] 之间

+ #### 公式

  $$
  x^{'} = \frac{x^{old}-min}{max-min}
  $$

  $$
  x^{new} = x^{'} * (mx - mi) + mi
  $$

  > max、min 分别为该特征数据中的最大值、最小值
  >
  > mx、mi 分别为设置的归一化区间的最大值、最小值

+ #### sklearn API：

  sklearn.preprocessing.MinMaxScaler (feature_range=(0,1))

  - MinMaxScalar.fit_transform(X)
    - X：numpy array 格式的数据
    - return：转换后的形状相同的array

+ #### 示例代码：

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('dating.txt')

# 1、创建传唤器，默认 feature_range=(0,1)
transfer = MinMaxScaler(feature_range=(2,3))

# 2、调用fit_transform
transfer.fit_transform(data[['milage','Liters','Consumtime','target']])
```

+ #### 归一化缺点：

  + 最大值最小值是变化的，归一化容易受极值影响，稳健性较差。
  + 最大值与最小值容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。

## 三、标准化

+ #### 定义

  通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内

+ #### 公式

  $$
  x^{new} = \frac{x - mean}{σ}
  $$

  >作用于每一列，mean为平均值，σ为标准差 


- #### sklearn API：

  sklearn.preprocessing.StandardScaler( )

  - 处理之后每列来说所有数据都聚集在均值0附近标准差差为1
  - StandardScaler.fit_transform(X)
    - X:numpy array格式的数据 
    - return：转换后的形状相同的array

- #### 示例代码：

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('dating.txt')

# 1、创建转换器
transfer = StandardScaler()

# 2、调用fit_transform
new_data = transfer.fit_transform(data[data.columns[:3]])

# 3、打印标准化后的结果
print(new_data)
```

+ #### 归一化的缺点在标准化下不存在

  + 对于标准化来说，如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小。
  + 在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。




























