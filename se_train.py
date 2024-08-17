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
code_file1 = './feature/{}/{}_EIIP_train.csv'.format(cell_name,cell_name)
code_file11 = './feature/{}/{}_3mer_train.csv'.format(cell_name,cell_name)
input_file1 = pd.read_csv(code_file1,header=None,index_col=[0])
input_file11 = pd.read_csv(code_file11,header=None,index_col=[0])

code_file5 = './feature/{}/train_{}4mer-3datavec.csv'.format(cell_name,cell_name)
input_file5 = pd.read_csv(code_file5,header=None,index_col=[0])
code_file7 = './feature/{}/train_{}5mer-3datavec.csv'.format(cell_name,cell_name)
input_file7 = pd.read_csv(code_file7,header=None,index_col=[0])
code_file9 = './feature/{}/train_{}6mer-3datavec.csv'.format(cell_name,cell_name)
input_file9 = pd.read_csv(code_file9,header=None,index_col=[0])


x_train = pd.concat([input_file1,input_file11],axis=1,ignore_index=True)
x_train = x_train.values
x_deep_train = pd.concat([input_file5,input_file7,input_file9],axis=1,ignore_index=True)
x_deep_train = x_deep_train.values

print("train形状",x_train.shape)


## 人类
y1_train = [1]*9307
y2_train = [0]*9307
y_train = y1_train + y2_train
y_train = np.array(y_train)

# ### 小鼠
# y1_train = [1]*10509
# y2_train = [0]*10509
# y_train = y1_train + y2_train
# y_train = np.array(y_train)

# 随机打乱索引顺序
shuffle_indices = np.random.permutation(len(x_deep_train))
x_train = x_train[shuffle_indices]
x_deep_train = x_deep_train[shuffle_indices]
y_train = y_train[shuffle_indices]

input_shape = (576, 1)
lstm_units = 64  # LSTM 单元数
attention_units = 64  # Attention 单元数
dropout_rate = 0.5  # 调整 Dropout 比率，根据需要
def lr_scheduler(epoch):
  if epoch < 10:
    return 0.001
  if epoch > 10 and epoch < 20:
    return 0.0005
  else:
    return 0.00005
lr_scheduler = LearningRateScheduler(lr_scheduler)
initial_learning_rate = 0.001  ##定义compile里的默认学习率，实际学习率会动态调整
num_epochs = 40

# 1.深度学习
x1_train = x_deep_train.reshape(x_deep_train.shape[0], 576, 1).astype('float32')  ##深度学习需要转化一下形状
# 创建数据加载器
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x1_train, y_train))
train_dataset = train_dataset.batch(batch_size)

model = build_binary_classification_model6(input_shape, lstm_units, attention_units, dropout_rate)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model.fit(train_dataset, epochs=num_epochs,callbacks=[lr_scheduler], verbose=2)
save_path = './model/deep_train.h5'
model.save(save_path)
print("完成深度学习训练任务1……")
##2.svm支持向量机
clf2 = SVC(kernel='rbf', C=1.35, probability=True, random_state=42)
clf2.fit(x_train, y_train)
joblib.dump(clf2, './model/svc_feature.joblib')
print("完成svc训练任务2……")
# 3.创建LightGBM分类器
lgb_classifier = lgb.LGBMClassifier(n_estimators=100,max_depth=10,random_state=42)
lgb_classifier.fit(x_train, y_train)
joblib.dump(lgb_classifier, './model/lgb_feature.joblib')
print("完成lightGBM训练任务3……")

