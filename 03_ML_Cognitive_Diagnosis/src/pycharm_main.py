COLOUR_FIGURE = True # 是否使用彩色绘图（如果为 False，则使用灰度图）

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加简单的KNN实现（替代缺失的knn模块）
def learn_model(k, features, labels):
    return (k, features, labels)

def apply_model(test_features, model):
    k, train_features, train_labels = model
    predictions = []
    for test_point in test_features:
        # 计算欧氏距离
        distances = np.sqrt(np.sum((train_features - test_point)**2, axis=1))
        # 获取最近的k个邻居
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        # 多数投票
        unique, counts = np.unique(nearest_labels, return_counts=True)
        prediction = unique[np.argmax(counts)]
        predictions.append(prediction)
    return np.array(predictions)

# 定义特征名称（数据集中每一列特征的含义）
feature_names = [
    '谷粒面积',  # 谷粒面积
    '周长',      # 周长
    '紧密度',    # 紧密度
    '谷粒长度',  # 谷粒长度
    '谷粒宽度',  # 谷粒宽度
    '不对称系数', # 不对称系数
    '谷粒沟长度' # 谷粒沟长度
]

def train_plot(features, labels):
    # 选择要可视化的两个特征：第0列(area) 和 第2列(compactness)，并确定它们的取值范围
    y0,y1 = features[:,2].min()*.9, features[:,2].max()*1.1
    x0,x1 = features[:,0].min()*.9, features[:,0].max()*1.1
    # 生成一个规则的二维网格
    X = np.linspace(x0,x1,100)
    Y = np.linspace(y0,y1,100)
    X,Y = np.meshgrid(X,Y)

    # 训练一个k=1的KNN模型，只使用特征0和2
    model = learn_model(1, features[:,(0,2)], np.array(labels))
    # 对网格中的每个点进行预测，得到对应的分类标签
    C = apply_model(np.vstack([X.ravel(),Y.ravel()]).T, model).reshape(X.shape)
    # 设置背景颜色（不同类别显示不同颜色）
    if COLOUR_FIGURE:
        cmap = ListedColormap([
            (1., 0.6, 0.6),  # 红色背景
            (0.6, 1., 0.6),  # 绿色背景
            (0.6, 0.6, 1.)   # 蓝色背景
        ])
    else:
        cmap = ListedColormap([
            (1., 1., 1.),    # 白色
            (0.2, 0.2, 0.2), # 深灰
            (0.6, 0.6, 0.6)  # 浅灰
        ])
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[2])
     # 在网格上绘制分类区域颜色
    plt.pcolormesh(X,Y,C, cmap=cmap)
    # 绘制样本点
    if COLOUR_FIGURE:
        # 使用不同颜色绘制三个类别的数据点
        cmap = ListedColormap([
            (1., 0., 0.),  # 红色
            (0., 1., 0.),  # 绿色
            (0., 0., 1.)   # 蓝色
        ])
        scatter = plt.scatter(features[:, 0], features[:, 2], c=labels, cmap=cmap)
        # 添加图例
        plt.legend(handles=scatter.legend_elements()[0], 
                  labels=['Kama', 'Rosa', 'Canadian'],
                  title="类别")
    else:
        # 如果是灰度模式，用不同标记符号表示不同类别
        for lab, ma in zip(range(3), "Do^"):  # D=菱形，o=圆形，^=三角形
            plt.plot(features[labels == lab, 0],
                     features[labels == lab, 2],
                     ma, c=(1., 1., 1.), label=f'类别{lab+1}')
        plt.legend(title="类别")

# 从本地文件加载数据
data = pd.read_csv('D:\gwzMLCode\seeds.tsv', sep='\s+', header=None, na_values='?')
data = data.dropna()  # 移除缺失值

# 分离特征和标签
features = data.iloc[:, :-1].values.astype(float)  # 前7列是特征
labels_str = data.iloc[:, -1].values  # 最后一列是字符串标签

# 将字符串标签转换为数字标签
unique_labels = sorted(set(labels_str))
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_to_num[label] for label in labels_str])

print(f"标签映射: {label_to_num}")
print(f"特征形状: {features.shape}")
print(f"标签形状: {labels.shape}")

train_plot(features, labels)
plt.title("原始数据 - KNN分类边界")
plt.show()

# 数据标准化
features -= features.mean(0)
features /= features.std(0)
train_plot(features, labels)
plt.title("标准化后数据 - KNN分类边界")
plt.show()