## 内容预览

- #### 1.1 深度学习与机器学习的区别

  - 1.1.1 特征提取方面
  - 1.1.2 数据量和计算性能要求
  - 1.1.3 算法代表

- #### 1.2 深度学习的应用场景

  - 1.2.1 图像识别
  - 1.2.2 自然语言处理技术
  - 1.2.3 语音技术

- #### 1.3 深度学习框架介绍

  - 1.3.1 常见深度学习框架对比
  - 1.3.2 TensorFlow的特点
  - 1.3.3 TensorFlow的安装

## 1.1 深度学习与机器学习的区别

+ #### 特征提取方面

  + 机器学习的特征工程要靠手动完成的，这需要大量的专业领域的知识
  + 深度学习由多层组成，它通常将更简单的模型组合在一起，将数据一层一层地传递来构建复杂的模型。通过大量数据训练自动得出模型，不需要人工特征提取环节。
  + 深度学习算法试图从数据中提取高级特征，这是深度学习的一个非常独特的部分。因此深度学习中不再为每个问题开发新特征提取器，这样深度学习更适用在**难提取特征**的图像、语音、自然语言处理领域。

- #### 数据量和计算性能要求

  深度学习需要的执行时间远大于机器学习，深度学习参数往往很庞大，需要通过大量数据的多次优化来训练参数。

  因此：

  1. 深度学习需要大量的训练数据集
  2. 训练深度神经网络需要大量的算力

  可能要花费数天、甚至数周的时间，才能使用数百万张图像的数据集训练出一个深度网络。

  所以深度学习通常：

  - 需要强大的GPU服务器来进行计算
  - 全面管理的分布式训练与预测服务——比如 [谷歌 TensorFlow 云机器学习平台](https://cloud.google.com/ml/)

- #### 算法代表
  - 机器学习
    - 朴素贝叶斯、决策树等
  - 深度学习
    - 神经网络

## 1.2 深度学习的应用场景

- #### 图像识别

  - 物体识别
  - 场景识别
  - 车型识别
  - 人脸检测跟踪
  - 人脸关键点定位
  - 人脸身份认证

- #### 自然语言处理技术

  - 机器翻译
  - 文本识别
  - 聊天对话

- #### 语音技术

  - 语音识别

## 1.3 深度学习框架介绍

+ #### 常见深度学习框架对比

  |   框架名   | 主语言 |       从语言       | 灵活性 | 上手难易 |      开发者      |
  | :--------: | :----: | :----------------: | :----: | :------: | :--------------: |
  | Tensorflow |  C++   |    cuda/python     |   好   |    难    |      Google      |
  |   Torch    |  Lua   |       C/cuda       |   好   |   中等   |     Facebook     |
  |  PyTorch   | Python |       C/C++        |   好   |   中等   |     Facebook     |
  |   Caffe    |  C++   | cuda/python/Matlab |  一般  |   中等   |      贾杨清      |
  |   Theano   | Python |      C++/cuda      |   好   |    易    | 蒙特利尔理工学院 |
  |   MXNet    |  C++   |    cuda/R/julia    |   好   |   中等   |  李沐和陈天奇等  |

  1. 最常用的框架当数 TensorFlow 和 Pytorch , Caffe 和 Caffe2 次之
  2. PyTorch , Torch 更适用于学术研究，TensorFlow，Caffe，Caffe2 更适用于工业界的生产环境部署
  3. Caffe 适用于处理静态图像；Torch 和 PyTorch 更适用于动态图像；TensorFlow 在两种情况下都很实用。
  4. Tensorflow 和 Caffe2 可在移动端使用。

+ #### TensorFlow 的特点

  官网：<https://www.tensorflow.org/>

  + 高度灵活
    - 它可以做神经网络算法，也可以做机器学习算法，甚至只要把计算表示成数据流图，都可以用TensorFlow 。
  + 语言多样
    - TensorFlow 使用 C++ 实现的，使用了 Python 进行封装。谷歌号召社区通过 SWIG 开发更多的语言接口来支持 TensorFlow 。
  + 设备支持
    - TensorFlow 可以运行在各种硬件上，同时根据计算的需要，合理将运算分配到相应的设备。比如卷积就分配到 GPU 上，也允许在 CPU 和 GPU 上的计算分布，甚至支持使用 gRPC 进行水平扩展。gRPC 是一个高性能、通用的开源RPC框架。
  + TensorBoard 可视化
    - TensorBoard 是 TensorFlow 的一组 Web 应用，用来监控 TensorFlow 运行过程 或 可视化Computation Graph。
    - TensorBoard 目前支持 5 种可视化：标量（scalars）、图片（images）、音频（audio）、直方图（histograms）和计算图（Computation Graph）。TensorBoard 的 Events Dashboard 可以用来持续地监控运行时的关键指标，比如 损失（loss）、学习速率（learning rate）或是 验证集上的准确率（accuracy）等等。

+ #### TensorFlow 的安装

  + ##### CPU版本

    安装较慢，最好指定镜像源，并在带有 numpy 等库的虚拟环境中安装

    ```python
    pip install tensorflow -i https://mirrors.aliyun.com/pypi/simple
    ```

  + ##### GPU版本

    ```python
    pip install tensorflow-gpu -i https://mirrors.aliyun.com/pypi/simple
    ```

  + ##### CPU与GPU的对比

    CPU：核芯的数量更少；但是每一个核芯的速度更快，性能更强；更适用于处理连续性（sequential）任务。

    GPU：核芯的数量更多；但是每一个核芯的处理速度较慢；更适用于并行（parallel）任务。



















