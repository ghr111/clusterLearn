"""
@author: liushuchun
"""
#整体思路： https://www.jb51.net/article/142541.htm
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 建立特征矩阵
# documents：是词袋列表
def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    # strip()默认去除首尾的空格
    feature_type = feature_type.lower().strip()
    # CountVectorizer用法：https://blog.csdn.net/weixin_38278334/article/details/82320307
    # CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率
    # @binary默认为False，一个关键词在一篇文档中可能出现n次，如果binary=True，非零的n将全部置为1，这对需要布尔值输入的离散概率模型的有用的
    # @max_df可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，
    # 如果某个词的document frequence大于max_df，这个词不会被当作关键词。
    # 如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效
    # @ngram_range = (1, 3) 表示选取 1 到 3 个词做为组合方式: 词向量组合为: 'I', 'like', 'you', 'I like', 'like you', 'I like you' 构成词频标签
    #选择不同的向量化策略这里是TF-IDF
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
    # fit_transform(x)	x是列表，每个列表包含一行文本，返回一个字典，key：词，value:频(？这里的词频为什么与输入的元素无关)
    # feature_matrix即tf_idf结果，fit_transform http://blog.sina.com.cn/s/blog_b8effd230102yznw.html
    # 数据归一化、标准化 https://www.cnblogs.com/pejsidney/p/8031250.html
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    # print("显示特征矩阵")
    # print(feature_matrix)
    # python返回多参数 https://www.cnblogs.com/wllhq/p/8119347.html
    return vectorizer, feature_matrix

from sklearn.cluster import KMeans

# k_means:https://blog.csdn.net/lynn_001/article/details/86679270
def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters, max_iter=10000)
    # k_means开始拟合分类
    km.fit(feature_matrix)
    # cluster存已经分好类的标签,是数字标明的列表，相同的数字代表是一类，和输入的wordList顺序一一对应
    clusters = km.labels_
    # print("--clusters--")
    # print(clusters)
    return km, clusters




from collections import Counter

# Counter用法：https://www.cnblogs.com/keke-xiaoxiami/p/8553076.html
# # 获取每个cluster的数量
# c = Counter(clusters)
# #以字典的形式显示里边的元素
# print('--显示特征元素--:')
# print(c.items())

# clustering_obj:keans对象，book_data：书的数据，feature_names：特征名字，num_clusters：要分多少类，topn_features：排序的前几类特征
def get_cluster_data(clustering_obj, book_data,
                     feature_names, num_clusters,
                     topn_features=10):
    cluster_details = {}
    # 获取cluster的center
    # 对得到的二维数组每个聚类中心点进行排序得到排序索引,这里"::-1"使其按照从大到小排序，ordered_centroids里存的是排序索引
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]

    # 获取每个cluster的关键特征
    # 获取每个cluster的书
    for cluster_num in range(num_clusters):
        # 声明字典
        cluster_details[cluster_num] = {}
        # 字典的key为cluster_num的值，value为一个字典->key为cluster_num值为cluster_num的值,{1: {'cluster_num': 1, 'key_features': with}}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        # key_features是包含多个关键字列表，对ordered_centroids二位数组进行遍历
        key_features = [feature_names[index]
                        for index in ordered_centroids[cluster_num, :topn_features]]

        cluster_details[cluster_num]['key_features'] = key_features
        # 把book_data字典中相同的cluster标记（代表是一类），它们是一类形成列表存到books中
        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books
    print('--cluster_details--')
    print(cluster_details)
    return cluster_details

# 显示分类结果
def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)



# ---client----

book_data = pd.read_csv('data/data.csv')  # 读取文件返回一个二维表格

# print('book_data')
# print(book_data.head())

# book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()#获取content即第书的内容，形成列表

# print('书名:', book_titles[0])
# print('内容:', book_content[0][:10])

from normalization import normalize_corpus

# normalize corpus 把列表中的中文语句转换成词组存到norm_book_content中，形成中文分词列表
norm_book_content = normalize_corpus(book_content)
print('--norm_book_content--')
print(norm_book_content)
# 提取 tf-idf 特征
# 获取TfidfVectorizer向量化构造器，获取特征矩阵
vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                  feature_type='tfidf',
                                                  min_df=0.2, max_df=0.90,
                                                  ngram_range=(1, 2))
# 查看特征矩阵
print('--查看特征矩阵规模 feature_matrix.shape--')
print(feature_matrix.shape)

# 获取特征名字
feature_names = vectorizer.get_feature_names()

# 打印某些特征,显示前10个
# print('--打印了后十个特征 feature_names[-10:]--')
# print(feature_names[-10:])

num_clusters = 10
# 放入特征矩阵，放入需要分类的数量，运行keans方法，获取KMeans对象，获取已经分好类的标签列表clusters
km_obj, clusters = k_means(feature_matrix=feature_matrix, num_clusters=num_clusters)
#databook是一个字典输出如下
book_data['Cluster'] = clusters
print('--book_data[Cluster]--')
print(book_data)
# 获取分好类的数据
cluster_data = get_cluster_data(clustering_obj=km_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=num_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

'''
#对分类结果进行画图
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib.font_manager import FontProperties

def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, book_data,
                  plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[0:500].items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': book_data['Cluster'].values.tolist(),
                                       'title': book_data['title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and book titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['title'], size=8)
        # show the plot
    plt.show()


plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              book_data=book_data,
              plot_size=(16, 8))
'''
'''
# 用AffinityPropagation算法来进行聚类
from sklearn.cluster import AffinityPropagation


def affinity_propagation(feature_matrix):
    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_
    return ap, clusters


# get clusters using affinity propagation
ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
book_data['Cluster'] = clusters

# get the total number of books per cluster
c = Counter(clusters)
print(c.items())

# get total clusters
total_clusters = len(c)
print('Total Clusters:', total_clusters)

cluster_data = get_cluster_data(clustering_obj=ap_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=total_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              book_data=book_data,
              plot_size=(16, 8))

from scipy.cluster.hierarchy import ward, dendrogram


def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


def plot_hierarchical_clusters(linkage_matrix, book_data, figure_size=(8, 12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    book_titles = book_data['title'].values.tolist()
    # plot dendrogram
    ax = dendrogram(linkage_matrix, orientation="left", labels=book_titles)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('ward_hierachical_clusters.png', dpi=200)


# build ward's linkage matrix
linkage_matrix = ward_hierarchical_clustering(feature_matrix)
# plot the dendrogram
plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                           book_data=book_data,
                           figure_size=(8, 10))
'''