import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi


# color:['Blues','BrBG','BuGn','BuPu','CMRmap','GnBu','Greens','Greys','OrRd','Oranges','PRGn','PiYG','PuBu','PuBuGn','PuOr','PuRd','Purples','RdBu','RdGy','RdPu','RdYlBu','RdYlGn','Reds','Spectral','Wistia','YlGn','YlGnBu','YlOrBr','YlOrRd','afmhot','autumn','binary','bone','brg','bwr','cool','coolwarm','copper','cubehelix','flag','gist_earth','gist_gray','gist_heat','gist_ncar','gist_rainbow','gist_stern','gist_yarg','gnuplot','gnuplot2','gray','hot','hsv','jet','nipy_spectral','ocean','pink','prism','rainbow','seismic','spring','summer','terrain','winter','Accent','Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3','tab10','tab20','tab20b','tab20c']


def 数学运算():
    print(1 + 1)
    print(1 - 1)
    print(1 * 1)
    print(1 / 1)
    print(1 % 1)
    print(1 ** 1)


def 逻辑运算():
    print(1 == 1)
    print(1 != 1)
    print(1 & 0)  # 与
    print(1 | 0)  # 或
    print(1 ^ 0)  # 异或


def 变量():
    a = 3
    b = 'hi'
    c = 3 > 1
    d = pi  # Π
    print(a)
    print(b)
    print(c)
    print('%.2f' % d)  # 保留两位小数
    print('%.6f' % d)  # 保留六位小数


def 矩阵():
    a = np.array([[1, 2], [3, 4], [5, 6]])  # 三行二列的矩阵a
    print(a)

    v = np.arange(3)  # 胖胖的 Y 向量( 1行3列的行向量 )
    print(v)

    v = v.reshape((3, 1))  # 列向量
    print(v)

    v = np.arange(10, 20, 2)  # 2 步长的 1 行 5 列的行向量
    print(v)

    ans = np.ones((2, 3), dtype=int)  # 所有元素都是 1
    print(ans)

    ans = 2 * ans  # 所有元素都是 2
    print(ans)

    ans = np.zeros((2, 3))  # 所有元素都是 0
    print(ans)

    w = np.random.random((1, 3))  # 生成以随机数组成的一行三列的矩阵,且所有元素介于0-1之间
    print(w)

    w = np.random.normal(size=(1, 3))  # 1行3列的高斯分布
    print(w)

    w = -6 + np.square(10) * np.random.random((1, 10000))  # 生成10000个元素
    print(w)

    ans = np.eye(3)  # 生成3×3单位矩阵
    print(ans)

    print(ans.size)  # 矩阵元素数量

    print(w[0, :10])  # 第1行,0到10列
    print(ans[:, 1])  # 所有行,第2列
    print(ans[[0, 2], :])  # 第1行和第3行,所有列
    print(ans[1, 2])  # 第2行第3列

    ans[:, 0] = [1, 1, 1]  # 取出第一列再赋值
    print(ans)

    a = np.arange(6).reshape((3, 2))
    b = np.arange(6, 12).reshape((3, 2))
    c = np.hstack((a, b))  # 左右合并
    print(c)
    d = np.vstack((a, b))  # 上下合并
    print(d)


def 矩阵运算():
    a = np.arange(6).reshape((3, 2))
    b = np.arange(6, 12).reshape((3, 2))
    c = np.array([[1, 1], [2, 2]])
    print(a)
    print(c)
    print(a.dot(c))  # 矩阵乘法 a×c
    print(a * b)  # 矩阵对应元素相乘
    print(a ** 2)  # 矩阵每个元素 平方
    print(1 / b)  # 矩阵每个元素 倒数
    print(np.log(b))  # 矩阵每个元素 log
    print(np.exp(b))  # 矩阵每个元素 e的元素方
    print(np.abs(b))  # 矩阵每个元素 绝对值
    print(-a)  # 矩阵每个元素 负数

    print(a + np.ones(np.shape(a)))  # 矩阵每个元素 加1
    print(a + 1)  # 矩阵每个元素 加1

    print(a.T)  # 矩阵转置
    print(a.T.T)  # 矩阵转置再转置

    print(a.max())  # 矩阵最大值
    print(a.max(axis=0))  # 矩阵每列取最大值
    print(a.min(axis=1))  # 矩阵每行取最小值
    print(np.where(a == a.max()))  # 矩阵中最大值索引 返回值前面对应行数,后面对应列数

    print(a < 3)  # 矩阵每个元素和3进行比较,返回一个布尔值矩阵
    print(a[a < 5])  # 返回矩阵中小于5的元素组成的列表

    print(a.sum())  # 矩阵每个元素相加
    print(a.prod())  # 矩阵每个元素相乘
    print(np.floor(a))  # 矩阵元素向下取整
    print(np.ceil(a))  # 矩阵元素向上取整

    # numpy没有魔方阵

    a = np.random.random((1, 3))
    b = np.random.random((1, 3))
    print(a)
    print(b)
    print(np.maximum(a, b))  # 两个矩阵的对应元素进行比较,取最大值组成新的矩阵

    a = np.arange(6).reshape((3, 2))
    print(a.argmax(axis=0))  # 返回每一列的最大值
    print(a.argmax(axis=1))  # 返回每一行的最大值

    a = np.arange(9).reshape((3, 3))
    print(a)
    print(np.sum(a * np.eye(3)))  # 正对角线之和
    print(np.sum(a * np.flipud(np.eye(3))))  # 副对角线之和, flipud表示向上翻转

    a = np.random.random((3, 3))
    print(np.linalg.inv(a))  # 逆矩阵


def 绘图():
    x = np.arange(0, 0.98, 0.01)
    y1 = np.sin(2 * pi * 4 * x)  # pi=数学里的pai
    y2 = np.cos(2 * pi * 4 * x)

    data1 = pd.DataFrame({
        'Y1': y1,
        'Y2': y2,
    })  # 只需要展示出y值的变化,因为y本身是随x变化的
    data2 = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list('ABCD')).cumsum()  # 结果是一个累加列表

    fig, axes = plt.subplots(2, 1)  # 创建2个并列在1行上的画板
    ax1 = data1.plot(title='mai', ax=axes[0])  # 画在第1个画板
    ax2 = data2.plot(title='mai', ax=axes[1])  # 画在第2个画板

    plt.xlabel('x')  # 设置x轴名称
    plt.ylabel('y')  # 设置y轴名称

    plt.xticks(np.arange(0, 1000, 200), np.arange(1000, 2000, 200))  # 修改x轴刻度 参数1旧刻度,参数2新刻度
    plt.show()  # 必须写这句,展示出来

    # 保存成图
    fig = ax1.get_figure()  # 获取所属fig
    fig.savefig('fig.png')


def 绘图矩阵():
    a = np.random.randn(5, 5)  # 定义矩阵
    img = plt.imshow(a, plt.get_cmap('gray'))  # 绘制colormap,color列表在文件开头
    # img = plt.imshow(a,cmap=cm.autumn)  # 和上面写法作用一样
    plt.colorbar(img)  # 添加一个colorbar
    plt.show()

def 循环():
    # 稍作改动使结果和视频一样
    v = np.zeros((10,1))
    for i in range(10):
        v[i] = 2**(i+1)
    print(v)

    i = 1
    while i<=5:
        v[i-1] = 100
        i += 1
    print(v)

    i = 1
    while True:
        v[i-1] = 999
        i += 1
        if i == 6:
            break
    print(v)

def 计算代价函数J():
    X = np.matrix([[1,1],[1,2],[1,3]])
    y = np.matrix([[1],[2],[3]])
    theta = np.matrix([[0],[1]])

    def Jdef(X,y,theta):  # 代价函数J
        m = np.size(X,axis=0)  # 训练集数目
        predict = X.dot(theta)
        sqrerrors = np.square(predict - y)
        J = 1/(2*m) * sum(sqrerrors)
        return J
    print('X:\n',X,'\n')
    print('y:\n',y,'\n')
    print('theta:\n',theta,'\n')
    print('J:',Jdef(X, y, theta))


if __name__ == '__main__':
    # 数学运算()
    # 逻辑运算()
    # 变量()
    # 矩阵()
    # 矩阵运算()
    # 绘图()
    # 绘图矩阵()
    # 循环()
    计算代价函数J()