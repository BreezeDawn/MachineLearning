## 一、特征降维概述

+ #### 为什么要对特征进行降维处理

  + 如果特征本身存在问题或者特征之间相关性较强，对于算法学习预测会影响较大


<!-- more -->

+ #### 什么是降维

  + 降维是指在某些限定条件下，降低随机变量(特征)个数，得到一组“不相关”主变量的过程

+ #### 降维的作用

  + 减少特征数量
  + 减少特征相关性，去除相关性强的特征，比如 相对湿度与降雨量 

+ #### 降维的两种方式

  + 特征选择
  + 主成分分析（PCA）



## 二、什么是特征选择

+ #### 定义

  旨在从原有特征中找出主要特征，去除冗余或无关特征。

+ #### 方法

  - Filter(过滤式)：主要探究特征本身特点、特征与特征和目标值之间关联。

    - 方差选择法：低方差特征过滤
    - 相关系数

  - Embedded (嵌入式)：算法自动选择特征（特征与目标值之间的关联）

    - 决策树:信息熵、信息增益
    - 正则化：L1、L2
    - 深度学习：卷积

    > Embedded方式，在讲解算法时再进行介绍

+ #### 模块

  ```python
  sklearn.feature_selection
  ```

## 三、降维 - 特征选择 - 过滤式 - 方差选择法

+ #### 低方差特征过滤，删除低方差的一些特征，

  + 特征方差小：在多个样本中某个特征的值会比较相近
  + 特征方差大：在多个样本中某个特征的值是有些许差别的

+ #### API

  sklearn.feature_selection.VarianceThreshold(threshold = 0.0)
  + 删除所有低方差特征
  + Variance.fit_transform(X)
    + X:numpy array格式的数据
    + 返回值：训练集方差 **低于** threshold 的特征将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征。

+ #### 示例代码

  ```python
  import pandas as pd
  from sklearn.feature_selection import VarianceThreshold
  
  data = pd.read_csv('factor_returns.csv')
  
  print(data[data.columns[1:-2]].shape)
  
  # 1、实例化一个转换器类
  transfer = VarianceThreshold(threshold=1)
  
  # 2、调用fit_transform
  new_data = transfer.fit_transform(data[data.columns[1:-2]])
  
  # 3、删除低方差特征的结果
  print(new_data.shape)
  ```


## 四、降维 - 特征选择 - 过滤式 - 相关系数

+ #### 皮尔逊相关系数(Pearson Correlation Coefficient)

  + 反映特征之间相关关系密切程度的统计指标

+ #### 公式(了解)

  $$
  r = \frac{n\sum{xy} - \sum{x}\sum{y}}{\sqrt{n\sum{x^2}-(\sum{x})^2}\sqrt{n\sum{y^2}-(\sum{y})^2}}
  $$

  > 上面是协方差，下面是各自的标准差

+ #### 特点

  + 相关系数的值介于 –1 与 +1 之间，即 $–1≤ r ≤+1$ 。
  + 当 $r>0$ 时，表示两变量正相关，$r<0$ 时，两变量为负相关
  + 当 0<|r|<1 时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱
  + 当|r|=1时，表示两变量为完全相关，当r=0时，表示两变量间无相关关系。
  + 一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关。

+ #### API

  from scipy.stats import pearsonr

+ #### 示例代码

  ```python
  import pandas as pd
  from scipy.stats import pearsonr
  
  data = pd.read_csv('./data/factor_returns.csv')
  
  factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
                'earnings_per_share', 'revenue', 'total_expense']
  
  datas = [(factor[i], factor[j + 1], pearsonr(data[factor[i]], data[factor[j + 1]])[0]) for i in range(len(factor)) for j in range(i, len(factor) - 1)]
  
  for data in datas:
      print("指标 {} 与指标 {} 之间的相关性大小为 {} ".format(*data))
  ```


## 五、降维 - 主成分分析（PCA）

+ #### 什么是主成分分析(PCA)

  + 定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量

  + 作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。

  + 应用：回归分析或者聚类分析当中

    > 在决策树中'信息'一词会有清晰理解

+ ### API

  sklearn.decomposition.PCA(n_components=None)

  - 将数据分解为较低维数空间
  - n_components:
    - 小数：保留百分之多少的信息
    - 整数：减少到具体的多少个特征
  - PCA.fit_transform(X) 
    - X：numpy array 格式的数据
    - return：转换为指定维度后的 array

+ #### 示例代码

  ```python
  from sklearn.decomposition import PCA
  
  data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
  
  # 1、实例化PCA, 小数—保留百分之多少信息
  transfer = PCA(n_components=0.9)
  # 2、调用fit_transform
  data1 = transfer.fit_transform(data)
  print("保留90%的信息，降维结果为：\n", data1)
  
  # 1、实例化PCA, 整数——指定降维到的维数
  transfer2 = PCA(n_components=3)
  # 2、调用fit_transform
  data2 = transfer2.fit_transform(data)
  print("降维到3维的结果：\n", data2)
  ```


## 六、降维 - 案例

+ #### 目的

  探究用户对物品类别的喜好细分降维

+ #### 现有数据

  + order_products__prior.csv：订单与商品信息
    - 字段：**order_id**, **product_id**, add_to_cart_order, reordered
  + products.csv：商品信息
    - 字段：**product_id**, product_name, **aisle_id**, department_id
  + orders.csv：用户的订单信息
    - 字段：**order_id**,**user_id**,eval_set,order_number,….
  + aisles.csv：商品所属具体物品类别
    - 字段： **aisle_id**, **aisle**

+ #### 分析
  + 合并表，使得**user_id**与**aisle**在一张表当中
  + 进行交叉表变换
  + 进行降维

+ #### 完整代码

  ```python
  import pandas as pd
  from sklearn.decomposition import PCA
  
  # 1、获取数据集 
  products = pd.read_csv("./data/instacart/products.csv")  # 商品信息
  order_products = pd.read_csv("./data/instacart/order_products__prior.csv")  # 订单与商品信息
  orders = pd.read_csv("./data/instacart/orders.csv")  # 用户的订单信息
  aisles = pd.read_csv("./data/instacart/aisles.csv")  # 商品所属具体物品类别
  
  # 2、合并表，将user_id和aisle放在一张表上
  # 1）合并 orders 和 order_products 
  tab1 = pd.merge(aisles, products, on="aisle_id")
  # 2）合并 tab1 和 products
  tab2 = pd.merge(tab1, order_products, on="product_id")
  # 3）合并 tab2 和 aisles 
  tab3 = pd.merge(tab2, orders, on="order_id")
  
  # 3、交叉表处理，把 user_id 和 aisle 进行分组 
  table = pd.crosstab(tab3["user_id"], tab3["aisle"])
  
  # 4、主成分分析的方法进行降维
  # 1）实例化一个转换器类PCA
  transfer = PCA(n_components=0.95)
  # 2）fit_transform
  data = transfer.fit_transform(table)
  
  # 查看降维结果
  data.shape
  ```