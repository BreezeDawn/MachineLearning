## 一、sklearn - 模型的保存和加载 - API

from sklearn.externals import joblib

- 保存
  - joblib.dump(rf, 'test.pkl')
- 加载
  - estimator = joblib.load('test.pkl')

<!-- more -->

## 二、示例助解

- #### 保存

  ```python
  # 使用线性模型进行预测
  # 使用正规方程求解
  lr = LinearRegression()
  # 进行训练
  lr.fit(x_train, y_train)
  # 保存训练完结束的模型
  joblib.dump(lr, "test.pkl")
  ```

- #### 加载

  ```python
  # 通过已有的模型去预测
  model = joblib.load("test.pkl")
  model.predict(x_test)
  ```
