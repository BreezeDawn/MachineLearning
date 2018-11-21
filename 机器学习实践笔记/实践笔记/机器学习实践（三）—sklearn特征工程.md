## 一、特征工程介绍

#### 1. 为什么需要特征工程

Andrew Ng ： “Coming up with features is difficult, time-consuming, requires expert knowledge. “Applied machine learning” is basically feature engineering. ”

> 注：业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。



#### 2. 什么是特征工程

特征工程是使用专业背景知识和技巧处理数据**，**使得特征能在机器学习算法上发挥更好的作用的过程。

- 意义：会直接影响机器学习的效果



#### 3. 特征工程包含的内容

- 特征抽取
- 特征预处理
- 特征降维



#### 4. 特征工程常用的模块

+ pandas：数据预处理，一个数据读取非常方便以及基本的处理格式的工具
  + 缺失值处理
  + 数据类型转换
  + 数据清洗
+ sklearn：特征工程，对于特征的处理提供了强大的接口
  + 特征提取
  + 特征预处理
  + 特征降维

## 二、特征提取

#### 1. 什么是特征提取

+ 将文本/字典/图像转换为数值
+ 设置哑变量，将类别型特征转换为 $0/1$ 格式

#### 2. 特征提取 API

+ sklearn.feature_extraction

## 三、字典特征提取

+ 作用：
  + 对字典数据进行特征值化
+ API：sklearn.feature_extraction.DictVectorizer(sparse=True)
  + DictVectorizer.fit_transform(X) 
    + X：字典或者包含字典的迭代器 
    + return：返回sparse矩阵
  + DictVectorizer.inverse_transform(X) 
    + X：array数组或者sparse矩阵
    + return：转换之前数据格式
  + DictVectorizer.get_feature_names() 
    + 返回类别名称

+ 示例代码：

```python
from sklearn.feature_extraction import DictVectorizer

data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]

# 实例化一个转换器类,sparse：True返回sparse矩阵，False 返回真实矩阵
transfer = DictVectorizer(sparse=True)
# 调用fit_transform
print(transfer.fit_transform(data))
# 打印特征名字
print(transfer.get_feature_names())	
```

## 四、文本特征提取

+ 作用：
  + 对文本数据进行特征值化

- sklearn.feature_extraction.text.CountVectorizer(stop_words=[])
  - 返回词频矩阵
  - CountVectorizer.fit_transform(X) 
    - X：文本或者包含文本字符串的可迭代对象 
    - return：返回sparse矩阵
  - CountVectorizer.inverse_transform(X) 
    - X：array数组或者sparse矩阵 
    - return：转换之前数据格
  - CountVectorizer.get_feature_names() 
    - return：单词列表

- sklearn.feature_extraction.text.TfidfVectorizer
  - tfidf 文本特征提取

+ 示例代码：

```python
from sklearn.feature_extraction.text import CountVectorizer

data = ["life is short,i like like python", "life is too long,i dislike python"]

# 实例化一个转换器类,stop_words->过滤单词 如：'is'¶
transfer = CountVectorizer(stop_words=['is'])
# 返回 sparse 矩阵，节省内存
print(transfer.fit_transform(data))
# 返回 真实矩阵，利用toarray()进行sparse矩阵转换array数组
print(transfer.fit_transform(data).toarray())
# 返回特征名字
print("返回特征名字：\n", transfer.get_feature_names())
```

## 五、中文文本特征提取

#### 1. 需要安装下jieba库

```python
pip3 install jieba
```

#### 2. jieba.cut()

- 返回词语组成的生成器

- 示例代码：

  - ```python
    text = '我爱北京天安门'
    print(' '.join(list(jieba.cut(text))))
    ```

### 3. 实例：中文文本特征提取

+ 示例代码：

```python
texts = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

# 创建分词匿名函数
cut_word = lambda text:' '.join(list(jieba.cut(text)))

# 对原始数据进行jieba分词
new_texts = [cut_word(text) for text in texts]

# 创建分类器
transfer = CountVectorizer()

# 返回频次数组
print(transfer.fit_transform(new_texts).toarray())

# 返回特征词
print(transfer.get_feature_names())
```



## 六、Tf-idf文本特征提取

+ #### TF-IDF的主要思想是：

  如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

+ #### TF-IDF作用：

  用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

+ #### 公式：


  $$
  tfidf_{ij} = tf_{ij} × idf_{ij}
  $$

  + > - 词频（term frequency，tf）
    >   - 指的是某一个给定的词语在该文件中出现的频率
    > - 逆向文档频率（inverse document frequency，idf）
    >   - 一个词语普遍重要性的度量。某一特定词语的idf，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到

+ 	

+ 词频（term frequency，tf）

  - 指的是某一个给定的词语在该文件中出现的频率

+ 逆向文档频率（inverse document frequency，idf）

  - 一个词语普遍重要性的度量。某一特定词语的idf，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到

+ #### 示例：

  >假如一篇文件的总词语数是100个，而词语"非常"出现了5次。
  >1. 那么"非常"一词在该文件中的词频就是5/100=0.05。
  >2. 如果"非常"一词在1,000份文件出现过，而文件总数是10,000,000份的话，其逆向文件频率就是lg（10,000,000 / 1,0000）=3。
  >3. 最后"非常"对于这篇文档的tf-idf的分数为0.05 * 3=0.15

+ #### 示例代码：

  ```python
  texts = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
              "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
              "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
  
  # 创建分词匿名函数
  cut_word = lambda text:' '.join(list(jieba.cut(text)))
  
  # 对原始数据进行jieba分词
  new_texts = [cut_word(text) for text in texts]
  
  # 创建分类器
  transfer = TfidfVectorizer(stop_words=['一种', '不会', '不要'])
  
  # 返回频次数组
  print(transfer.fit_transform(new_texts).toarray())
  
  # 返回特征词
  print(transfer.get_feature_names())
  ```


































































