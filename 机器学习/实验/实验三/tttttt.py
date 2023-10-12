import scipy.io
import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 读取.mat文件
data = scipy.io.loadmat('D:\code\大三课程\机器学习\实验\实验三\cats.vs.dogs.mat')

cats_data = data['cats']
dogs_data = data['dogs']

# 可视化图像数据
Num_cats = cats_data.shape[3]
rows_cat = 10
cols_cat = int(Num_cats/rows_cat)

Num_dogs = dogs_data.shape[3]
rows_dog = 10
cols_dog = int(Num_cats/rows_dog)

# 创建一个列表，用于存储每个图像
cats = [cats_data[:, :, :, i] for i in range(Num_cats)]
dogs = [dogs_data[:, :, :, i] for i in range(Num_dogs)]

# 使用OpenCV的cv2.vconcat和cv2.hconcat来创建montage
montage_cats = cv2.vconcat([cv2.hconcat(cats[i:i+cols_cat]) for i in range(0, len(cats), cols_cat)])
montage_dogs = cv2.vconcat([cv2.hconcat(dogs[i:i+cols_dog]) for i in range(0, len(dogs), cols_dog)])

# 显示猫和狗的montage图像
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(montage_cats, cmap='gray')
ax1.set_title('猫的图像')
ax1.axis('off')

ax2.imshow(montage_dogs, cmap='gray')
ax2.set_title('狗的图像')
ax2.axis('off')

plt.show()



import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns

cats = np.array(cats)
dogs = np.array(dogs)

# 将数据重塑为 (样本数量, 特征数量)
cats_data = cats.reshape(80, -1)
dogs_data = dogs.reshape(80, -1)

# 创建标签向量，1 表示猫，-1 表示狗
cat_labels = np.ones(80)
dog_labels = -np.ones(80)

# 合并猫和狗的数据和标签
X = np.vstack((cats_data, dogs_data))
y = np.hstack((cat_labels, dog_labels))


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 创建一个PCA对象并设置要降到的维度
pca = PCA(n_components=2)

# 对数据进行PCA降维
X_pca = pca.fit_transform(X)


# 定义SVM模型
svm_model = svm.SVC()

# 定义要搜索的参数网格
# 网格搜索
param_grid = {
    'C': [0.1, 1, 10],                    # 正则化参数
    'kernel': ['linear', 'rbf', 'poly'],  # 核函数
    'degree': [2, 3],                 # 多项式核的阶数
    'gamma': ['scale', 0.1, 1, 10],      # RBF核的尺度参数
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(svm_model, param_grid)

# 执行网格搜索
grid_search.fit(X_pca, y)

# 打印最佳参数和对应的准确率
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# 获取最佳模型
best_svm = grid_search.best_estimator_

# 训练最佳模型
best_svm.fit(X_pca, y)

# 计算训练集准确率
y_train_pred = best_svm.predict(X_pca)
train_accuracy = accuracy_score(y, y_train_pred)
print("Train Accuracy:", train_accuracy)

# 绘制决策边界和支持向量
def plot_svm_decision_boundary(svm_model, title, X, y):
    plt.figure(figsize=(8, 6))
    # 绘制样本点，用不同的颜色和标记表示猫和狗
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='猫')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='狗')

    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # 绘制支持向量
    plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k', label='支持向量')

    plt.title(title)
    plt.legend()
    plt.show()

# 绘制决策边界和支持向量
plot_svm_decision_boundary(best_svm, "Best SVM 分类边界")