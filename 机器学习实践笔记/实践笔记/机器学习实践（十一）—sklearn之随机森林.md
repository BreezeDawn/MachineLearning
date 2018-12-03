## 一、什么是集成学习方法

集成学习通过建立几个模型组合的来解决单一预测问题。

它的工作原理是生成多个分类器/模型，各自独立地学习和作出预测。

这些预测最后结合成组合预测，因此优于任何一个单分类的做出预测。

<!-- more -->

## 二、什么是随机森林

在机器学习中，随机森林是一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定。

例如, 如果你训练了5个树, 其中有4个树的结果是True, 1个数的结果是False, 那么最终投票结果就是True

## 三、随机森林原理过程

学习算法根据下列算法而建造每棵树：

- 用N来表示训练用例（样本）的个数，M表示特征数目。
  - 1 一次随机选出一个样本，重复N次， （有可能出现重复的样本）
  - 2 随机去选出m个特征, m <<M，建立决策树
- 采取bootstrap抽样

### 四、为什么采用BootStrap抽样

- 为什么要随机抽样训练集？　　
  - 如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的
- 为什么要有放回地抽样？
  - 如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是“有偏的”，都是绝对“片面的”（当然这样说可能不对），也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决。

## 五、sklearn - 随机森林 - API

- class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
  - 随机森林分类器
  - n_estimators：
    - integer，optional（default = 10）
    - 森林里的树木数量 120,200,300,500,800,1200
  - criteria：
    - string，可选（default =“gini”）
    - 分割特征的测量方法
  - max_depth：
    - integer或None，可选（默认=无）
    - 树的最大深度 5,8,15,25,30
  - max_features="auto”,每个决策树的最大特征数量
    - If "auto", then `max_features=sqrt(n_features)`.
    - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.
  - bootstrap：
    - boolean，optional（default = True）
    - 是否在构建树时使用放回抽样
  - min_samples_split：
    - 节点划分最少样本数
  - min_samples_leaf：
    - 叶子节点的最小样本数
  - 超参数：
    - n_estimator
    - max_depth
    - min_samples_split
    - min_samples_leaf

## 六、案例 - 随机森林 - 随机森林预测tanic生存状况¶

+ #### 完整代码

  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV
  
  # 1> 实例化一个估计器
  estimator=RandomForestClassifier()
  
  # 2> 网格搜索优化随机森林模型
  param_dict={"n_estimators":[120,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}
  estimator=GridSearchCV(estimator,param_grid=param_dict,cv=5)
  
  # 3> 传入训练集，进行模型训练
  estimator.fit(x_train,y_train)
  
  # 4> 模型评估
  # 方法1，比较真实值与预测值
  y_predict=estimator.predict(x_test)
  print("预测值为:\n",y_predict)
  print("比较真实值与预测值结果为:\n",y_predict==y_test)
  # 方法2,计算模型准确率
  print("模型准确率为:\n",estimator.score(x_test,y_test))
  print("在交叉验证中最的结果:\n",estimator.best_score_)
  print("最好的参数模型:\n",estimator.best_estimator_)
  print("每次交叉验证后的结果准确率为/n",estimator.cv_results_)
  ```

## 七、随机森林的优缺点

+ #### 优点

  + 不用作数据预处理
  + 不用作特征工程
  + 较好的效果和准确率

+ #### 缺点

  + 超参数优化时比较耗时


