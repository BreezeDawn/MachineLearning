import numpy as np


def base_test():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    print('矩阵:', arr)  # 矩阵
    print('类型:', arr.dtype)  # 类型
    print('维数:', arr.ndim)  # 维数
    print('行数/列数:', arr.shape)  # 行数/列数
    print('元素个数:', arr.size)  # 元素个数

    arr = np.zeros((3, 4))  # 所有元素为0的矩阵
    print('所有元素为0:', arr)
    arr = np.ones((3, 4), dtype=int)  # 所有元素为1的矩阵
    print('所有元素为1:', arr)
    arr = np.empty((3, 4))  # 所有元素近似于0的矩阵
    print('所有元素近似于0:', arr)
    arr = np.arange(10, 20, 2)  # 按步长生成范围数
    print('按步长生成范围数:', arr)
    arr = np.arange(12).reshape((3, 4))  # 生成12范围数并按3行4列排
    print('生成12范围数并按3行4列排:', arr)
    arr = np.linspace(10, 20, 2).reshape((2, 1))  # 从start开始到end结束的范围内生成num个数,可以reshape
    print('生成num个数:', arr)


def add_and_so_on():
    a = np.array([10, 20, 30, 40])  # 加减乘除直接算/乘并不是矩阵的乘法,而是单个相乘
    b = np.arange(4)
    print(a + b)

    print(b ** 2)  # 幂

    c = 10 * np.sin(a)  # sin
    print(c)

    print(b < 3)  # 结果为布尔值组成的列表


def dot_and_so_on():
    a = np.array([[10, 20], [30, 40]])
    b = np.arange(4).reshape((2, 2))
    # 矩阵乘法
    c = np.dot(a, b)  # 第一种方式
    print('乘法第一式:', c)
    d = a.dot(b)  # 第二种方式
    print('乘法第二式:', d)

    a = np.random.random((2, 4))  # 随机生成0-1的数并组成矩阵
    print(a)
    a = np.arange(14, 2, -1).reshape((3, 4))
    print('最大值:', np.max(a, axis=0))  # 最大值, axis: 1表示按行分组计算 0表示按列分组计算
    print('最大值索引:', np.argmax(a))  # 最大值下标
    print('最小值:', np.min(a, axis=1))  # 最小值
    print('最小值索引:', np.argmin(a))  # 最小值下标
    print('总和:', np.sum(a, axis=0))  # 总和
    print('平均值:', np.mean(a, axis=0))  # 平均值第一式
    print('平均值:', a.mean())  # 平均值第二式
    print('中位数:', np.median(a))  # 中位数
    print('逐步累加:', np.cumsum(a))  # 逐步累加生成列表
    print('相邻差:', np.diff(a))  # 每行中两两之差
    print('排序:', np.sort(a))  # 排序
    print('转置:', np.transpose(a))  # 转置第一式
    print('转置:', a.T)  # 转置第二式
    print('归数:', np.clip(a, 5, 9))  # 小于5的都等于5,大于9的都等于9
    print('下标取值:', a[1][2])  # 下标索引取值
    print('步长取值:', a[1, 0:2])  # 步长索引取值
    for raw in a:  # 行迭代
        print(raw)
    for column in a.T:  # 列迭代
        print(column)
    print(a.flatten())  # 矩阵变为一行
    for item in a.flat:  # 矩阵变为一行的迭代器
        print(item)


def merge_test():
    """合并"""
    A = np.array([1, 1, 1])
    B = np.array([2, 2, 2])
    C = np.vstack((A, B))  # 上下合并
    D = np.hstack((A, B))  # 左右合并
    print(C, C.shape)
    print(D, D.shape)
    A = A[:, np.newaxis]  # 冒号在前行上加维/冒号在后列上加维
    B = B[:, np.newaxis]
    print(np.vstack((A, B)))
    print(np.hstack((A, B)))
    print(np.concatenate((A, B, A, B), axis=1))  # axis: 1合并成行/左右合并 0合并成列/上下合并

def Segmentation_test():
    """分割"""
    A = np.arange(12).reshape(3,4)
    print(np.split(A, indices_or_sections=2, axis=1))  # 等项分割(分割块数需要合理)axis: 1按列分2块 0按行分2块
    print(np.array_split(A, 3,axis=1))  # 不等项分割,可以把4列按列分成3块,第一块2列,其他一列
    print(np.vsplit(A,3))  # 按行分割/上下分割
    print(np.hsplit(A,2))  # 按列分割/左右分割

def copy_test():
    a = np.arange(4)
    b = a
    c = a
    d = b
    print(b is a, c is a, d is a)
    # 上面四个全是a,改变任意一个,四个都会发生改变

    e = a.copy()
    print(e is a)
    # 使用copy方法,实际是deepcopy,e - a 互不影响


if __name__ == '__main__':
    base_test()
    # add_and_so_on()
    dot_and_so_on()
    # merge_test()
    # Segmentation_test()
    # copy_test()