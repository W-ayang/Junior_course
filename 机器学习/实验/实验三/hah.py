import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, (2, 3)]
y = iris.target
setosa_or_versicolor = (y == 1) | (y == 2)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
svm_clf = SVC(kernel='linear', C=1e9)
svm_clf.fit(X, y)

# 绘制SVM决策边界
def plot_svc_decision_boundary(svm_clf, xmin, xmax, sv=True):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    if sv:
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='none', edgecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, 'k-', linewidth=2)
    plt.plot(x0, gutter_up, 'k--', linewidth=2)
    plt.plot(x0, gutter_down, 'k--', linewidth=2)

plt.figure(figsize=(7, 4))
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'g^')
plt.axis([0, 5.5, 0, 2])
plt.title("SVM分类器的决策边界")
plt.show()

# 软间隔SVM
iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 2).astype(np.float64)

svm_clf = Pipeline([
    ('std', StandardScaler()),
    ('linear_svc', LinearSVC(C=1))
])
svm_clf.fit(X, y)

# 创建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVM模型并生成混淆矩阵
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix_heatmap(confusion_matrix, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

# 绘制混淆矩阵图
plt.figure(figsize=(6, 6))
plot_confusion_matrix_heatmap(confusion, "Linear SVM 混淆矩阵")
plt.show()

# 高斯核函数SVM
X, y = make_moons(n_samples=50, noise=0.25, random_state=79)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'g^')
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

gamma_values = [0.2, 10]
C_values = [0.01, 1000]

svm_clfs = []

for gamma in gamma_values:
    for C in C_values:
        rbf_kernel_svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm_clf', SVC(kernel='rbf', gamma=gamma, C=C))
        ])
        rbf_kernel_svm_clf.fit(X, y)
        svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11, 7))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = gamma_values[i // len(C_values)], C_values[i % len(C_values)]
    plt.title(r'$σ = {}, C = {}$'.format(gamma, C), fontsize=16)

plt.suptitle("不同 σ 和 C 时SVM的表现", fontsize=20)
plt.show()

# 交叉验证寻找最佳参数
gamma_values = [0.2, 10]
C_values = [0.01, 1000]

cv_scores = []

for gamma in gamma_values:
    for C in C_values:
        rbf_kernel_svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm_clf', SVC(kernel='rbf', gamma=gamma, C=C))
        ])
        scores = cross_val_score(rbf_kernel_svm_clf, X, y, cv=5, scoring='accuracy')
        cv_scores.append((gamma, C, scores.mean()))

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
    plt.title(r'交叉验证的最佳参数：$ σ = {}, C = {}$'.format(best_gamma, best_C), fontsize=16)

plt.show()
