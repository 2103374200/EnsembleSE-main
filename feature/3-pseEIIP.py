import csv
import pandas as pd
import numpy as np
import re
# 定义存储 DNA 序列和标签的列表


file_name = "train_human.csv"
# 指定 CSV 文件路径
csv_file_path = "../data/datasets/"+ file_name  # 请替换成你的 CSV 文件路径

# 打开 CSV 文件并读取数据
with open(csv_file_path, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)
    p_sequence = []
    n_sequence = []
    for row in csv_reader:
        if 'N' not in row[0]:
            if row[1] == '0' :
                n_sequence.append(row[0])
            if row[1] == '1' :
                p_sequence.append(row[0])
        else:
            if row[1] == '0' :
                row[0] = row[0].replace('N', '')
                n_sequence.append(row[0])
            if row[1] == '1' :
                row[0] = row[0].replace('N', '')
                p_sequence.append(row[0])

    sequence = []
    sequence = p_sequence + n_sequence
    print(len(p_sequence))
    print(len(n_sequence))
    print(len(sequence))
    # print(sequence)
## 特征编码！
###3mer字典
kmer_list = []
for base1 in "ACGT":
    for base2 in "ACGT":
        for base3 in "ACGT":
            kmer = base1 + base2 +base3
            kmer_list.append(kmer)
###3merEIIP列表
kEIIP = []
for kmer in kmer_list:
    tmp = 0
    for i in range(0,3):
        if kmer[i] == 'A':
            tmp = tmp + 0.126
        if kmer[i] == 'G':
            tmp = tmp + 0.0806
        if kmer[i] == 'C':
            tmp = tmp + 0.1340
        if kmer[i] == 'T':
            tmp = tmp + 0.1335
    kEIIP.append(tmp)

def cal_3mer_freq(seq):
    kmer_freq = []
    for kmer in kmer_list:
        num_kmer = 0
        for i in range(0, len(seq) - 2):
            if kmer == seq[i:i+3]:
                num_kmer = num_kmer + 1
        kmer_freq.append(num_kmer/(len(seq)-2))
    return kmer_freq

def func(sequences):
    results = []
    result1s = []
    result2s = []
    result3s = []
    for seq in sequences:
        midpoint = int(len(seq) / 3)
        mid2point = int((len(seq)-midpoint)/2 + midpoint)
        first_half = seq[0:midpoint]
        second_half = seq[midpoint:mid2point]
        third_half = seq[mid2point:len(seq)]
        first_half = cal_3mer_freq(first_half)
        second_half = cal_3mer_freq(second_half)
        third_half = cal_3mer_freq(third_half)
        temp = 0
        result2 = []
        result1 = []
        result3 = []
        for i in range(0, 64):
            temp = kEIIP[i] * first_half[i]
            result1.append(temp)
        result1s.append(result1)
        for i in range(0, 64):
            temp = kEIIP[i] * second_half[i]
            result2.append(temp)
        result2s.append(result2)
        for i in range(0, 64):
            temp = kEIIP[i] * third_half[i]
            result3.append(temp)
        result3s.append(result3)
    results = [x + y + z for x, y, z in zip(result1s, result2s, result3s)]
    print(len(results))
    print(len(results[0]))
    return results

###执行编码
sequence_vector = func(sequence)
sequence_array = np.array(sequence_vector)
print(sequence_array)
# #

df = pd.DataFrame(sequence_array.reshape(sequence_array.shape[0], -1))  # 将多维数组转为二维DataFrame
df.to_csv('./human/human_EIIP_train.csv', index=True, header=False)  # 保存到CSV文件，保存行索引和不保存列头
print("++")