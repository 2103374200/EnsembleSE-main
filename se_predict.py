import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.svm import SVC
import numpy as np
from model.alstmSE import build_binary_classification_model6
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
# 设置全局随机种子
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
def z_score_normalize(datas):
    normalized_datas = []
    for data in datas:
        normalized_data = []
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        for i in range(0, len(data)):
            temp = (data[i] - mean) / std_dev
            normalized_data.append(temp)
        normalized_datas.append(normalized_data)
    normalized_datas = np.array(normalized_datas)
    return normalized_datas
cell_name = 'human'
## 加载特征矩阵，标准化
code_file2 = './feature/{}/{}_EIIP_test.csv'.format(cell_name,cell_name)
input_file2 = pd.read_csv(code_file2,header=None,index_col=[0])
code_file22 = './feature/{}/{}_3mer_test.csv'.format(cell_name,cell_name)
input_file22 = pd.read_csv(code_file22,header=None,index_col=[0])

code_file6 = './feature/{}/test_{}4mer-3datavec.csv'.format(cell_name,cell_name)
input_file6 = pd.read_csv(code_file6,header=None,index_col=[0])
code_file8 = './feature/{}/test_{}5mer-3datavec.csv'.format(cell_name,cell_name)
input_file8 = pd.read_csv(code_file8,header=None,index_col=[0])
code_file10 = './feature/{}/test_{}6mer-3datavec.csv'.format(cell_name,cell_name)
input_file10 = pd.read_csv(code_file10,header=None,index_col=[0])




x_EIIP_test = input_file2.values
x_3mer_test = input_file22.values
x_deep_test = pd.concat([input_file6,input_file8,input_file10],axis=1,ignore_index=True)
x_deep_test = x_deep_test.values
x_test = pd.concat([input_file2,input_file22],axis=1,ignore_index=True)  ### EIIP+3mer
x_test = x_test.values


x_EIIP_test = z_score_normalize(x_EIIP_test)

x_3mer_test = z_score_normalize(x_3mer_test)
x_deep_test = z_score_normalize(x_deep_test)


## 人类
y1_test = [1]*1034
y2_test = [0]*1034
y_test = y1_test+y2_test
y_test = np.array(y_test)
# ### 小鼠
# y1_test = [1]*1168
# y2_test = [0]*1168
# y_test = y1_test+y2_test
# y_test = np.array(y_test)

# 1.深度学习
x1_test = x_deep_test.reshape(x_deep_test.shape[0], 576, 1).astype('float32')
model_path = './model/deep_train.h5'  # 加载训练好的模型
loaded_model = load_model(model_path)
y1_pred = loaded_model.predict(x1_test)
print("完成深度学习预测任务1……")
##2.svm支持向量机
loaded_svc_model = joblib.load('./model/svc_feature.joblib')
probs = loaded_svc_model.predict_proba(x_test)
y2_pred = probs[:, 1]
print("完成svc预测任务2……")
# 3.创建LightGBM分类器
loaded_lgb_model = joblib.load('./model/lgb_feature.joblib')
probs = loaded_lgb_model.predict_proba(x_test)
y2_pred = probs[:, 1]
print("完成lightGBM预测任务3……")

##加权平均
y_pred = []
for i in range(0,len(y1_pred)):
    temp = (y1_pred[i][0] + y2_pred[i] +y3_pred[i]) / 3
    y_pred.append(temp)
y_pred = np.array(y_pred)  ##格式转化
# 计算各种性能指标
auc = roc_auc_score(y_test, y1_pred)
acc = accuracy_score(y_test, (y1_pred > 0.5).astype(int))
recall = recall_score(y_test, (y1_pred > 0.5).astype(int))
precision = precision_score(y_test, (y1_pred > 0.5).astype(int))
f1 = f1_score(y_test, (y1_pred > 0.5).astype(int))
aupr = average_precision_score(y_test, y1_pred)
# 输出各个折的性能指标
print("AUC:", auc)
print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("AUPR:", aupr)

