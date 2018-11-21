import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def base():
    index = pd.date_range('20181023', periods=9)  # 生成9个行索引
    column = ['a', 'b', 'c', 'd']  # 生成4个列索引
    a = np.random.randn(9, 4)  # 随便生成的9行4列的数据
    df = pd.DataFrame(a, index=index, columns=column)
    print(df)
    print(pd.DataFrame(np.arange(9).reshape((3, 3))))  # 行和列的默认索引为从0开始的数字
    print(df.dtypes)  # 查看每列的数据类型
    print(df.index)  # 查看每行的行索引
    print(df.columns)  # 查看每列的列索引
    print(df.values)  # 查看所有值
    print(df.describe())  # 查看每列的详细统计  数目/平均值/....
    print(df.T)  # pandas的转置
    print(df.sort_index(axis=1, ascending=False))  # 按索引排序 axis: 1列排序 0行排序 ascending: False反排序(从小向大) True正排序(从大向小)
    print(df.sort_values(by='a'))  # 把a列的值进行排序 默认从小向大


def select():
    index = pd.date_range('20181023', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=index, columns=['A', 'B', 'C', 'D'])
    print(df)
    print(df.A)  # 取出A列数据(带索引)
    print(df[2:3])  # 切片取数据
    print(df[2:3])  # 切片取数据
    print(df['2018-10-25':'2018-10-26'])  # 切片取数据
    print(df.loc['2018-10-25', ['A', 'B']])  # 按照标签取数据
    print(df.iloc[[1, 3, 5], 1:5])  # 按照数字取数据
    print(df.ix['2018-10-25':'2018-10-26', 1:5])  # 数字标签结合取数据
    print(df[df.A > 8])  # A列中的元素大于8的都显示


def update():
    index = pd.date_range('20181023', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=index, columns=['A', 'B', 'C', 'D'])
    df.iloc[2, 3] = -555  # 修改值  选中就能修改
    df.B[df.A > 8] = 0  # A列中的元素大于8的都把B修改为0
    print(df)
    df['E'] = pd.Series(np.arange(6), pd.date_range('20181023', periods=6))  # 增加一列
    print(df)


def handle_NaN():
    index = pd.date_range('20181023', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=index, columns=['A', 'B', 'C', 'D'])
    df.iloc[1, 2] = np.nan
    df.iloc[0, 1] = np.nan
    print(df)
    print(df.dropna(axis=1, how='any'))  # 丢掉缺失值(返回新的结果不影响原始数据) axis: 1丢掉列 0丢掉行  how: any任何一个是NaN就丢掉 all全是NaN就丢掉
    print(df.fillna(value=0))  # 填充缺失值 填充为0
    print(df.isnull())  # 检查每个元素是否缺失值,结果返回一个bool填充
    print(np.any(df.isnull()))  # np.any 检查至少有一个False,是的话返回True


def read_save_data():
    data = pd.read_csv('./pand.csv')  # 读取csv文件数据(csv内部逗号分隔)
    print(data)
    data.to_pickle('./pand.pickle')  # 保存数据到pickle文件


def merge_DataFrame():
    df1 = pd.DataFrame(np.zeros((3, 4)), columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)), columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(2 * np.ones((3, 4)), columns=['a', 'b', 'c', 'd'])
    print(df1)
    print(df2)
    print(df3)
    res = pd.concat([df1, df2, df3], axis=0)  # axis: 0上下合并 1左右合并
    print(res)
    res = pd.concat([df1, df2, df3], axis=1, ignore_index=True)  # ignore_index 忽略前面所有的index并重新排序
    print(res)

    df1 = pd.DataFrame(np.zeros((3, 4)), columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
    df2 = pd.DataFrame(np.ones((3, 4)), columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
    res = pd.concat([df1, df2], axis=0, join='outer', sort=True)  # 上下合并,outer如果有不一样的列字段,就用NaN填充
    print(res)
    res = pd.concat([df1, df2], axis=0, join='inner', sort=True, ignore_index=True)  # 上下合并, inner有不一样的列字段就丢掉那一列,保留相同字段
    print(res)
    res = pd.concat([df1, df2], axis=1, )  # 左右合并,有不一样的行字段就用NaN填充
    print(res)
    res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])  # 左右合并,行字段按照df1的行字段来,缺失值用NaN填充,其余df1没有的字段丢掉
    print(res)

    df1 = pd.DataFrame(np.zeros((3, 4)), columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)), columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)), columns=['a', 'b', 'c', 'd'])
    res = df1.append(df2, ignore_index=True)  # df1后面加上df2
    print(res)
    res = df1.append([df2, df3], ignore_index=True)  # df1后面加上df2,df3
    print(res)
    sl = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    res = df1.append(sl, ignore_index=True)
    print(res)


def merge():
    left = pd.DataFrame({
        'key': ['K0', 'K1', 'K2', 'K3'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    })
    right = pd.DataFrame({
        'key': ['K0', 'K1', 'K2', 'K3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    })
    print(left)
    print(right)
    res = pd.merge(left, right, on='key')  # 左右合并,key字段保留一个
    print(res)

    left = pd.DataFrame({
        'key1': ['K0', 'K0', 'K1', 'K2'],
        'key2': ['K0', 'K1', 'K0', 'K1'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    })
    right = pd.DataFrame({
        'key1': ['K0', 'K1', 'K1', 'K2'],
        'key2': ['K0', 'K0', 'K0', 'K0'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    })
    res = pd.merge(left, right, on=['key1', 'key2'], how='inner')  # 解释不清,看结果
    print(res)
    res = pd.merge(left, right, on=['key1', 'key2'], how='outer',indicator='indicator_column')  # 不管一不一样都保留 indicator写出哪些一样哪些不一样,写字符串可改名
    print(res)
    res = pd.merge(left, right, on=['key1', 'key2'], how='left')  # 左的on字段完全不动的保留
    print(res)
    res = pd.merge(left, right, on=['key1', 'key2'], how='right')  # 右的on字段完全不动的保留
    print(res)
    res = pd.merge(left, right, left_index=True,right_index=True, how='right')  # 根据索引保留
    print(res)


def plot_test():
    # 1000个一维数据累加
    data = pd.Series(np.random.randn(1000),index=np.arange(1000))
    data = data.cumsum()
    # data.plot()
    # plt.show()

    # 矩阵
    data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list('ABCD'))
    data = data.cumsum()
    print(data.head())  # head显示前五个数据,默认5个
    data.plot()  # 线性
    ax = data.plot.scatter(x='A',y='B',color='DarkBlue', label='Class 1')  # scatter 数据点 只有x,y
    data.plot.scatter(x='A',y='C',color='DarkGreen', label='Class 2',ax=ax)  # ax和前面的在一张图上
    plt.show()
    # plot method : bar条形图 hist box kde area scatter hexbin pie




if __name__ == '__main__':
    # base()
    # select()
    # update()
    # handle_NaN()
    # read_save_data()
    # merge_DataFrame()
    # merge()
    plot_test()