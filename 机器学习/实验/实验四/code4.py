import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 寻找最佳K值
def find_optimal_k(data, max_k,ispro):
    """
    使用肘部法则确定K-Means算法的最佳K值。
    参数:
    data (numpy.ndarray): 包含数据的数组，每行表示一个数据点。
    max_k (int): 尝试的最大K值。

    返回:
    int: 最佳的K值。
    """
    inertias = []  # 用于存储不同K值对应的簇内平方和

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        if ispro == True:
            kmeans.fit_pro(data)
        else:
            kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    
    optimal_k = 1
    for k in range(1, max_k):
        if inertias[k] - inertias[k - 1] < 0.2 * inertias[0]:  # 根据需要调整阈值
            optimal_k = k + 1
            break
    

    # 绘制K值与簇内平方和之间的关系图
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='-', color='b')
    plt.xlabel('k值数量')
    plt.ylabel('聚合成本（簇内平方和）')
    plt.title('聚合系数折线图')
    plt.grid(True)
    # 标记肘部的位置
    plt.annotate('肘部',
                xy=(optimal_k, inertias[optimal_k - 1]),
                xytext=(2, 100),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=12, color='red')
    plt.show()

    return optimal_k

# 寻找最佳K值
def find_optimal_k_silhouette(data, max_k,ispro):
    """
    使用轮廓系数确定K-Means算法的最佳K值。
    参数:
    data (numpy.ndarray): 包含数据的数组，每行表示一个数据点。
    max_k (int): 尝试的最大K值。

    返回:
    int: 最佳的K值。
    """
    silhouette_scores = []  # 用于存储不同K值对应的轮廓系数

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        if ispro == True:   
            kmeans.fit_pro(data)
        else:
            kmeans.fit(data)
        score = kmeans.silhouette_score_
        silhouette_scores.append(score)

    optimal_k = 2 + np.argmax(silhouette_scores)  # 因为K从2开始计算，所以要加2
    max_silhouette_score = max(silhouette_scores)

    # 绘制K值与轮廓系数之间的关系图
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('k值数量')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数折线图')
    plt.grid(True)
    # 标记最佳K值的位置
    plt.annotate(f'最佳K值: {optimal_k}\n轮廓系数: {max_silhouette_score:.2f}',
                 xy=(optimal_k, max_silhouette_score),
                 xytext=(6, 0.5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=12, color='red')
    plt.show()

    return optimal_k
    
class KMeans:
    """
    K-Means 聚类算法的实现。

    参数：
    n_clusters (int): 聚类的数量，默认为 2。
    max_iters (int): 最大迭代次数，默认为 100。
    random_state (int): 随机种子，用于初始化中心点的随机性。默认为 None。
    
    属性：
    labels_ (numpy.ndarray): 每个样本的聚类标签。
    cluster_centers_ (numpy.ndarray): 聚类中心的坐标。
    inertia_ (float): 损失函数值。
    silhouette_score_ (float): 轮廓系数值。
    """
    def __init__(self, n_clusters=2, max_iters=1000, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.silhouette_score_ = None  # 添加 silhouette_score_ 属性
    
    def fit(self, X):
        """
        使用 K-Means 算法拟合数据。

        参数：
        X (numpy.ndarray): 输入数据，每行表示一个样本，每列表示一个特征。

        返回值：
        self: 返回当前的 KMeans 实例。
        """
        np.random.seed(self.random_state)

        # 随机初始化中心点
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        inertia_list = []
        start_time = time.time()  # 记录开始时间

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(self.cluster_centers_ == new_centers):
                break
            self.cluster_centers_ = new_centers
            inertia = self.compute_inertia(X)
            inertia_list.append(inertia)  # 记录每次迭代的Inertia

        end_time = time.time()  # 记录结束时间

        self.inertia_ = self.compute_inertia(X)
        self.silhouette_score_ = self.compute_silhouette_score(X)

        self.iterations = _ + 1  # 迭代步数
        self.convergence_time = end_time - start_time  # 收敛时间

        # 可视化每次迭代的Inertia
        plt.plot(range(1, self.iterations), inertia_list, marker='o', linestyle='-')
        plt.xlabel("迭代次数")
        plt.ylabel("簇内平方和")
        plt.title("K-means迭代情况")
        plt.grid(True)
        plt.show()


        return self


    def fit_pro(self, X):
        """
        使用 K-Means 算法拟合数据。

        参数：
        X (numpy.ndarray): 输入数据，每行表示一个样本，每列表示一个特征。

        返回值：
        self: 返回当前的 KMeans 实例。
        """
        np.random.seed(self.random_state)
        
        # 使用 K-means++ 初始化中心点
        self.cluster_centers_ = self.initialize_centers(X)
        
        inertia_list = []
        start_time = time.time()  # 记录开始时间
        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(self.cluster_centers_ == new_centers):
                break
            self.cluster_centers_ = new_centers
            inertia = self.compute_inertia(X)
            inertia_list.append(inertia)  # 记录每次迭代的Inertia
        
        end_time = time.time()  # 记录结束时间

        self.inertia_ = self.compute_inertia(X)
        self.silhouette_score_ = self.compute_silhouette_score(X)

        self.iterations_pro = _ + 1  # 迭代步数
        self.convergence_time_pro = end_time - start_time  # 收敛时间
        # 可视化每次迭代的Inertia
        plt.plot(range(1, self.iterations_pro), inertia_list, marker='o', linestyle='-')
        plt.xlabel("迭代次数")
        plt.ylabel("簇内平方和")
        plt.title("K-means迭代情况")
        plt.grid(True)
        plt.show()

        return self
    
    def initialize_centers(self, X):
        """
        使用 K-means++ 初始化聚类中心。

        参数：
        X (numpy.ndarray): 输入数据，每行表示一个样本，每列表示一个特征。

        返回值：
        numpy.ndarray: 初始化的聚类中心坐标。
        """
        centers = [X[np.random.choice(len(X))]]  # 随机选择第一个中心点
        
        for _ in range(1, self.n_clusters):
            # 计算每个数据点到最近中心点的距离的平方
            distances = np.min(np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2)**2, axis=1)
            
            # 按概率选择下一个中心点
            probabilities = distances / np.sum(distances)
            next_center_index = np.random.choice(len(X), p=probabilities)
            centers.append(X[next_center_index])
        
        return np.array(centers)
    
    def compute_inertia(self, X):
        """
        计算簇内平方和，即损失函数值。

        参数：
        X (numpy.ndarray): 输入数据，每行表示一个样本，每列表示一个特征。

        返回值：
        float: 簇内平方和。
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            center = self.cluster_centers_[k]
            cluster_distance = np.linalg.norm(cluster_points - center, axis=1)
            inertia += np.sum(cluster_distance ** 2)
        return inertia
    
    def compute_silhouette_score(self, X):
        """
        计算轮廓系数。

        参数：
        X (numpy.ndarray): 输入数据，每行表示一个样本，每列表示一个特征。

        返回值：
        float: 轮廓系数值。
        """
        n = len(X)
        silhouette_scores = np.zeros(n)

        for i in range(n):
            # 计算样本i所属的簇
            cluster_i = self.labels_[i]
            
            # 计算a(i)，即样本i到同簇其他样本的平均距离
            cluster_i_points = X[self.labels_ == cluster_i]
            a_i = np.mean(np.linalg.norm(X[i] - cluster_i_points, axis=1))
            
            b_i = float('inf')  # 初始化b(i)为正无穷
            
            # 计算b(i)，即样本i到其他簇的平均最短距离
            for cluster in range(self.n_clusters):
                if cluster != cluster_i:
                    cluster_j_points = X[self.labels_ == cluster]
                    b_ij = np.mean(np.linalg.norm(X[i] - cluster_j_points, axis=1))
                    b_i = min(b_i, b_ij)
            
            # 计算轮廓系数，如果分母为零，则将轮廓系数设置为零
            if max(a_i, b_i) == 0:
                silhouette_scores[i] = 0
            else:
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        return np.mean(silhouette_scores)

if __name__ == "__main__":
    
    data_df = pd.read_csv('机器学习\实验\实验四\iris.csv')
    X = data_df.iloc[1:,1:5].to_numpy()
    
    # 创建标准化器
    scaler = StandardScaler()

    # 对数据进行标准化
    X_scaled = scaler.fit_transform(X)

    data_df = pd.read_csv('机器学习\实验\实验四\iris.csv')
    features = data_df.iloc[1:,1:5]

    # 设置图形大小
    plt.figure(figsize=(12, 6))
    boxprops = {'color': 'blue','facecolor': 'pink','alpha': 0.5}
    # 绘制箱线图
    plt.boxplot(features.values, vert=False, labels=features.columns,boxprops=boxprops,patch_artist=True)
    # 设置图形标题和标签
    plt.title('鸢尾花特征箱线图')
    plt.xlabel('Value')
    plt.ylabel('Feature')
    # 显示图形
    plt.show()


    # 计算四分位数和IQR
    Q1 = features.quantile(0.25)
    Q3 = features.quantile(0.75)
    IQR = Q3 - Q1

    # 计算异常值的上界限和下界限
    Upper_Bound = Q3 + 1.5 * IQR
    Lower_Bound = Q1 - 1.5 * IQR

    # 标识异常值
    outliers = ((features < Lower_Bound) | (features > Upper_Bound)).any(axis=1)

    # 打印异常值
    print("异常值的行：")
    print(features[outliers])

    # 去除异常值
    cleaned_data = features[~outliers]
    plt.boxplot(cleaned_data.values, vert=False, labels=cleaned_data.columns,boxprops=boxprops,patch_artist=True)
    # 设置图形标题和标签
    plt.title('鸢尾花特征箱线图(去除异常值后)')
    plt.xlabel('Value')
    plt.ylabel('Feature')
    # 显示图形
    plt.show()


    # 使用肘部法则选取最佳的k值
    max_k = 15
    k = find_optimal_k(X_scaled, max_k,False)
    k_pro = find_optimal_k(X_scaled, max_k,True)
    # 使用轮廓系数选择最佳的K值
    max_k_silhouette = 15  # 设置尝试的最大K值
    k_silhouette = find_optimal_k_silhouette(X_scaled, max_k_silhouette,False)
    k_silhouette_pro = find_optimal_k_silhouette(X_scaled, max_k_silhouette,True)
    # 降到二维进行可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 绘制降维后的散点图
    colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
    color_labels = data_df['Species'][1:].map(colors)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_labels, cmap='viridis', s=50)
    plt.title("鸢尾花数据集降维后的散点图")
    
    # 添加标签
    plt.annotate('Setosa', xy=(X_pca[:, 0].max() + 0.1, X_pca[:, 1].max() + 0.1), fontsize=12, color='red', ha='center', va='center')
    plt.annotate('Versicolor', xy=(X_pca[:, 0].max() + 0.1, X_pca[:, 1].max() - 0.1), fontsize=12, color='blue', ha='center', va='center')
    plt.annotate('Virginica', xy=(X_pca[:, 0].max() + 0.1, X_pca[:, 1].max() - 0.3), fontsize=12, color='green', ha='center', va='center')

    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.grid(True)
    
    plt.show()

    # 3. 运行K-means(传统kmeans)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    
    plt.figure(figsize=(8, 6))
    for cluster_label in set(kmeans.labels_):
        cluster_points = X_pca[kmeans.labels_ == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, label=f'第 {cluster_label + 1} 类')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='聚类中心')
    plt.title("K-Means 聚类")
    plt.legend(loc='upper right')
    plt.show()

    print(f'轮廓系数为：{kmeans.silhouette_score_}')
    print(f'损失函数值：{kmeans.inertia_}')

    ########################################################
    # kmeans++
    # 3. 运行K-means(kmeans++)
    kmeans = KMeans(n_clusters=k_pro, random_state=42)
    kmeans.fit_pro(X_pca)
    
    plt.figure(figsize=(8, 6))
    for cluster_label in set(kmeans.labels_):
        cluster_points = X_pca[kmeans.labels_ == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, label=f'第 {cluster_label + 1} 类')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='聚类中心')
    plt.title("K-Means 聚类")
    plt.legend(loc='upper right')
    plt.show()


    print(f'轮廓系数为：{kmeans.silhouette_score_}')
    print(f'损失函数值：{kmeans.inertia_}')