import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, average_precision_score
import pandas as pd
import matplotlib.pyplot as plt

code1 = 'mouse_EIIP_train.csv'
input_file1 = pd.read_csv(code1, header=None, index_col=[0])
x_EIIP = input_file1.values
print(len(x))

code2 = 'mouse_3mer_train.csv'
input_file2 = pd.read_csv(code2, header=None, index_col=[0])
x_3mer = input_file2.values
### 小鼠
y1_train = [1]*10509
y2_train = [0]*10509
y_train = y1_train + y2_train
y_train = np.array(y_train)
y = y_train
## 人类
# y1_train = [1]*9307
# y2_train = [0]*9307
# y_train = y1_train + y2_train
# y_train = np.array(y_train)
# y = y_train
# y1_test = [1]*1168
# y2_test = [0]*1168
# y_test = y1_test+y2_test
# y_test = np.array(y_test)
###############1
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x_EIIP, y, test_size=0.2, random_state=42,stratify=y)

# 初始化 SVM 分类器
svm_classifier = SVC(probability=True)
num_features_range = range(10, 192, 10)
acc_scores = []

for num_features in num_features_range:
    # 计算每个特征的 F 分数
    f_scores, _ = f_classif(X_train, y_train)
    # 获取 F 分数对应的列的索引，并按 F 分数从高到低排序
    selected_indices = np.argsort(f_scores)[::-1][:num_features]
    # 选择相应数量的特征
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    # 训练 SVM 模型
    svm_classifier.fit(X_train_selected, y_train)
    # 预测概率
    y_prob = svm_classifier.predict_proba(X_test_selected)[:, 1]
    # 计算准确率
    acc = accuracy_score(y_test, y_prob > 0.5)
    acc_scores.append(acc)
    print(f"Num Features: {num_features}, Accuracy: {acc:.4f}")
###############################2
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x_3mer, y, test_size=0.2, random_state=42,stratify=y)
# 初始化 SVM 分类器
svm_classifier = SVC(probability=True)
num_features_range_2 = range(10, 192, 10)
acc_scores_2 = []
for num_features in num_features_range_2:
    # 计算每个特征的 F 分数
    f_scores, _ = f_classif(X_train, y_train)
    # 获取 F 分数对应的列的索引，并按 F 分数从高到低排序
    selected_indices = np.argsort(f_scores)[::-1][:num_features]
    # 选择相应数量的特征
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    # 训练 SVM 模型
    svm_classifier.fit(X_train_selected, y_train)
    # 预测概率
    y_prob = svm_classifier.predict_proba(X_test_selected)[:, 1]
    # 计算准确率
    acc = accuracy_score(y_test, y_prob > 0.5)
    acc_scores_2.append(acc)
    print(f"Num Features: {num_features}, Accuracy: {acc:.4f}")

# 绘制平滑曲线图
# plt.plot(num_features_range, acc_scores, marker='o', label='PseEIIP')
# plt.plot(num_features_range_2, acc_scores_2, marker='o', label='3mer_freq')
# plt.show()