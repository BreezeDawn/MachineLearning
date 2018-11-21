---
title: 机器学习-向量化
tags: 机器学习
categories: 机器学习
abbrlink: 16142
date: 2018-10-25 10:37:09
---

### 向量化 - 传统累加运算 - 代码实现:

```python
import time
import numpy as np

# 定义两组向量
vector1 = np.random.rand(100000)
vector2 = np.random.rand(100000)

# 使用向量化
start_time = time.time()  # 开始时间
res = np.dot(vector1, vector2)  # 向量直接相乘得到最终结果
end_time = time.time()  # 结束时间
print("Vectorized: " + str((end_time - start_time)*1000) + "ms" + " res =" + str(res))

# 使用for循环
res = 0
start_time = time.time()  # 开始时间
for i in range(100000):  # 传统的累加运算,需要累加100000次
    res += vector1[i] * vector2[i]
end_time = time.time()  # 结束时间
print("For loop: " + str((end_time - start_time)*1000) + "ms" + " res =" + str(res))
```

<!-- more -->

### 结果对比:

```python
Vectorized :1.0001659393310547ms res =24969.775960643143
For loop:79.94818687438965ms res =24969.775960642968
```

​        从执行结果来看向量化的运算速度要比非向量化的运算快了近80倍，而这个对比结果还会随着运算集的数目增加而增加。

### 为什么:

​        CPU 与 GPU 都能够使用 SIMD 指令进行并行化操作，即以同步方式，在同一时间内执行同一条指令。一般来讲可扩展的深度学习都在 GPU 上做，但其实 CPU 也不是太差，只是没有 GPU 擅长。

​        而 Python 的 numpy 的一些内置函数能够充分利用并行化来加速运算，比如 np.dot，因此，不到逼不得已，还是不要使用 for 循环吧

### 注:

​        GPU - 图形处理器也，叫做图像处理单元，显卡的处理器。与 CPU 类似，只不过 GPU 是专为执行复杂的数学和几何计算而设计的，这些计算是图形渲染所必需的。
​        SIMD - 单指令多数据流，以同步方式，在同一时间内执行同一条指令。