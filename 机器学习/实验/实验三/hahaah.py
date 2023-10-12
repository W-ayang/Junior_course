import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import scipy.io
from sklearn.model_selection import cross_val_score

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# 鸢尾花 数据集
iris = datasets.load_iris()
X = iris['data'][:,(2, 3)]
y = iris['target']
setosa_or_versicolor = (y==0)|(y==1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
svm_clf = SVC(kernel='linear', C = 1e9)
svm_clf.fit(X, y)


# SVM 的决策边界
def plot_svc_decision_boundary(svm_clf, xmin, xmax, sv = True):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    if sv:
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:,0], svs[:,1], s=180, facecolors='red', edgecolors='k')
    plt.plot(x0, decision_boundary, 'k-', linewidth=2)
    plt.plot(x0, gutter_up, 'k--', linewidth=2)
    plt.plot(x0, gutter_down, 'k--', linewidth=2)

plt.figure(figsize = (7,4))
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:,0][y==1], X[:,1][y==1], 'bs')
plt.plot(X[:,0][y==0], X[:,1][y==0], 'rs')
plt.axis([0, 5.5, 0, 2])
plt.title("Linear_SVM分类器超平面")
plt.show()


# 软间隔
# (C 较小时会让决策边界不那么严格，防止过拟合)
iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
y = (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline((
    ('std', StandardScaler()),
    ('linear_svc', LinearSVC(C=1))
))
svm_clf.fit(X, y)

svm_clf.predict([[5.5, 1.7]])

scaler = StandardScaler()
scaler.fit(X)

svm_clf1 = LinearSVC(C=0.1, random_state = 42)
svm_clf2 = LinearSVC(C=100, random_state = 42)

scaled_svm_clf1 = Pipeline((
    ('std', scaler),
    ('linear_svc', svm_clf1)
))

scaled_svm_clf2 = Pipeline((
    ('std', scaler),
    ('linear_svc', svm_clf2)
))
scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)

b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'ro', label = 'Iris-Virginica')
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs', label = 'Iris-Versicolor')
plot_svc_decision_boundary(svm_clf1, 4, 6, sv=False)
plt.xlabel('Petal length', fontsize=14)
plt.ylabel('Petal width', fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.title('$C = {}$'.format(svm_clf1.C), fontsize=16)
plt.axis([4, 6, 0.8, 2.8])
plt.show()

plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'ro')
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
plot_svc_decision_boundary(svm_clf2, 4, 6, sv=False)
plt.xlabel('Petal length', fontsize=14)
plt.title('$C = {}$'.format(svm_clf2.C), fontsize=16)
plt.axis([4, 6, 0.8, 2.8])
plt.show()


# 高斯核函数
X, y = make_moons(n_samples=50, noise=0.25, random_state=79)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'ro')
    plt.axis(axes)
    plt.grid(True, which='both')

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title("训练数据")
plt.show()

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)

gamma_values = [0.1, 1, 10]
C_values = [0.1, 10, 100]

hyperparams = [(gamma, C) for gamma in gamma_values for C in C_values]

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', gamma=gamma, C=C))
    ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11, 7))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(331 + i)  # 修改这里的子图索引以匹配新的组合
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r'$\gamma = {}, C = {}$'.format(gamma, C), fontsize=16)

plt.suptitle("不同 γ 和 C 时SVM效果", fontsize=20)
plt.show()

gamma_v = [0.1, 10,100]
C_v = [0.01, 1,100,1000]
# 创建空列表来存储交叉验证的分数
cv_scores = []

# 遍历所有可能的组合
for gamma in gamma_v:
    for C in C_v:
        rbf_kernel_svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm_clf', SVC(kernel='rbf', gamma=gamma, C=C))
        ])
        # 使用交叉验证计算分数
        scores = cross_val_score(rbf_kernel_svm_clf, X, y, cv=5, scoring='accuracy')
        # 计算平均分数并将其存储在cv_scores中
        cv_scores.append((gamma, C, scores.mean()))

# 找到具有最高平均分数的组合
best_gamma, best_C, best_score = max(cv_scores, key=lambda x: x[2])

svm_clfs = []
rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=best_gamma, C=best_C))
])
rbf_kernel_svm_clf.fit(X, y)
svm_clfs.append(rbf_kernel_svm_clf)

for i, svm_clf in enumerate(svm_clfs):
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r'交叉验证的最佳参数：$\gamma = {}, C = {}$'.format(best_gamma, best_C), fontsize=16)

plt.show()