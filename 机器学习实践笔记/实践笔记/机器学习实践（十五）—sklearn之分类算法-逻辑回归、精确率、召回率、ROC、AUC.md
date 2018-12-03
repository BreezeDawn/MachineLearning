逻辑回归虽然名字中带有回归两字，但它实际是一个分类算法。

<!-- more -->

## 一、逻辑回归的应用场景

- 广告点击率
- 是否为垃圾邮件
- 是否患病
- 金融诈骗
- 虚假账号

看到上面的例子，我们可以发现其中的特点，那就是都属于两个类别之间的判断。

逻辑回归就是解决二分类问题的利器

## 二、逻辑回归的原理

+ #### 输入

  逻辑回归的输入其实就是线性回归

  即：
  $$
  h_\theta(x)=\theta^Tx
  $$

+ #### 激活函数(sigmoid)

  逻辑回归的本质就是把输入到线性回归产生的结果再输入到激活函数中然后输出。

  即：
  $$
  g(h_\theta(x)) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}
  $$
  **输出的结果为：[0,1]区间中的一个概率值，默认的阈值为 0.5。**

  > 如：假设有两个类别A，B，并且我们认为阈值为0.5，输出结果超过阈值则预测为 A 类。那么现在有一个样本的输入到逻辑回归输出结果 0.6，这个概率值超过 0.5，意味着我们训练或者预测的结果就是A类别。那么反之，如果得出结果为 0.3 那么，训练或者预测结果就为B类别。

## 三、损失以及优化

那么如何去衡量逻辑回归的预测结果与真实结果的差异呢？

+ #### 损失

  逻辑回归的损失，称之为**对数似然损失**，公式如下：
  $$
  Cost(h_\theta(x),y) = -ylog(h_\theta(x)) - (1-y)log(1-h_\theta(x))
  $$
  上式为针对单条数据的损失函数，

  那么，我们能够得出总的损失函数，公式如下：
  $$
  \sum_{i=1}^{m} y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)})
  $$

+ #### 优化

  我们同样可以使用梯度下降优化算法，去减少损失函数的值。

  这样去更新逻辑回归前面对应算法的权重参数，提升原本属于1类别的概率，降低原本是0类别的概率。

## 四、逻辑回归API

+ sklearn.linear_model.LogisticRegression(solver='liblinear', penalty=‘l2’, C = 1.0)
  + solver
    - 优化求解方式（默认开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数）
    - 默认使用的是 sag，即根据数据集自动选择，随机平均梯度下降
  + penalty
    - 正则化的种类
    - 默认为 L2
  + C
    - 正则化力度
+ 使用 SGDClassifier 实现逻辑回归 API，SGDClassifier(loss="log", penalty=" ")
  + SGDClassifier实现了一个普通的随机梯度下降学习，也可以通过设置average=True，实现随机平均梯度下降。
  + loss，设置 log ，即逻辑回归中的对数损失函数

## 五、案例：癌症分类预测-良／恶性乳腺癌肿瘤预测

+ #### 数据介绍

  原始数据的下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

  > 下载数据：breast-cancer-wisconsin.data

+ #### 数据描述

  + 699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤

  相关的医学特征，最后一列表示肿瘤类型的数值。

  + 包含16个缺失值，用 ”?” 标出。

+ #### 步骤分析

  - 缺失值处理
  - 标准化处理
  - 逻辑回归预测

+ #### 完整代码

  ```python
  import pandas as pd
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  
  # 获取数据并添加字段名
  column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                     'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                     'Normal Nucleoli', 'Mitoses', 'Class']
  cancer=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",names=column_name)
  cancer.head()
  
  # 缺失值处理
  cancer=cancer.replace(to_replace="?",value=np.nan)
  cancer=cancer.dropna()
  
  # 数据集划分
  # 1> 提取特征数据与目标数据
  x=cancer.iloc[:,1:-2]
  y=cancer.iloc[:,-1]
  # 2> 划分数据集
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
  
  # 标准化处理
  transfer=StandardScaler()
  x_train=transfer.fit_transform(x_train)
  x_test=transfer.transform(x_test)
  
  # 模型训练
  # 创建一个逻辑回归估计器
  estimator=LogisticRegression()
  # 训练模型，进行机器学习
  estimator.fit(x_train,y_train)
  # 得到模型，打印模型回归系数，即权重值
  print("logist回归系数为:\n",estimator.coef_)
  
  # 模型评估
  # 方法1：真实值与预测值比对
  y_predict=estimator.predict(x_test)
  print("预测值为:\n",y_predict)
  print("真实值与预测值比对:\n",y_predict==y_test)
  # 方法2：计算准确率
  print("直接计算准确率为:\n",estimator.score(x_test,y_test))
  ```


## 六、二分类 - 模型评估 - 精确率、召回率 与 $F_1-score$ 

+ #### 混淆矩阵

  在分类任务下，预测结果与正确标记之间存在四种不同的组合，构成混淆矩阵(适用于多分类)

  |            | 预测为正例 | 预测为假例 |
  | :--------: | :--------: | :--------: |
  | 真实为正例 | 真正例(TP) | 伪反例(FN) |
  | 真实为假例 | 伪正例(FP) | 真反例(TN) |

+ #### 精确率

  $$
  \frac{真正例}{预测为正例}
  $$

+ #### 召回率 - (查得全，对正样本的区分能力)

  $$
  \frac{真正例}{真实为正例}
  $$

+ #### $F_1 - score$ - (反映了模型的稳健型)

  $$
  F_1 = \frac{2TP}{2TP+FP+FN}
  $$

+ #### 分类评估报告 - API

  sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None )

  + y_true
    - 真实目标值
  + y_pred
    - 估计器预测值
  + labels
    - 指定类别对应的数字
  + target_names
    - 目标类别名称
  + return
    - 每个类别精确率、召回率、$F_1$ 系数以及该类占样本数 

  如：`classification_report(y_test, lr.predict(x_test), labels=[2, 4], target_names=['良性', '恶性'])`

## 七、二分类 - 模型评估 - ROC曲线与AUC指标

+ #### 如何衡量样本不均衡下的评估？

​	假设这样一个情况，如果99个样本癌症，1个样本非癌症，不管怎样我全都预测正例(默认癌症为正例),准确率就为99%但是这样效果并不好，这就是样本不均衡下的评估问题。

+ #### TPR

  $$
  TPR = \frac{TP}{TP + FN}
  $$

  > 真实为真时预测为真 占 真实为真的 比例

+ #### FPR

  $$
  FPR = \frac{FP}{FP + TN}
  $$

  >  真实为假时预测为真 占 真实为假的 比例

+ #### ROC曲线

  ROC 曲线的横轴就是FPR，纵轴就是TPR

  ![](\img\ROC.png)

+ #### AUC指标

  - AUC 的概率意义是随机取一对正负样本，正样本得分大于负样本的概率
  - AUC 的最小值为0.5，最大值为1，取值越高越好
  - AUC=1，完美分类器，采用这个预测模型时，不管设定什么阈值都能得出完美预测。绝大多数预测的场合，不存在完美分类器。
  - 0.5<AUC<1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。

+ #### AUC - API

  from sklearn.metrics import roc_auc_score

  - sklearn.metrics.roc_auc_score(y_true, y_score)
    - 计算ROC曲线面积，即AUC值
    - y_true
      - 每个样本的真实类别，必须为0(反例),1(正例)标记
    - y_score
      - 预测得分，可以是正类的估计概率、置信值或者分类器方法的返回值
    - return
      - AUC值

+ #### 关于AUC

  - AUC只能用来评价二分类
  - AUC非常适合评价样本不平衡中的分类器性能

## 八、案例 - 精确率、召回率、AUC值

```python
# 接上面的肿瘤预测代码

#打印精确率、召回率、F1 系数以及该类占样本数
print("精确率与召回率为:\n",classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶性"]))

###模型评估
#ROC曲线与AUC值
# 把输出的 2 4 转换为 0 或 1
y_test=np.where(y_test>2,1,0)  # 大于2就变为1，否则变为0
print("AUC值:\n",roc_auc_score(y_test,y_predict))
```









