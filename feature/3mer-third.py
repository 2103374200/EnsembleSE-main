import csv
import pandas as pd
import numpy as np

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

## 特征编码
def calculate_3mer_frequency(sequence):
    # 初始化3mer频率字典
    frequency_dict = {}
    # 遍历序列，计算3mer频率
    for i in range(len(sequence) - 2):
        kmer = sequence[i:i + 3]
        if kmer in frequency_dict:
            frequency_dict[kmer] += 1
        else:
            frequency_dict[kmer] = 1

    return frequency_dict

def generate_3mer_vector(sequences):
    # 按照A，G，C，T的顺序创建3mer列表
    kmer_list = []
    for base1 in "AGCT":
        for base2 in "AGCT":
            for base3 in "AGCT":
                kmer = base1 + base2 + base3
                kmer_list.append(kmer)
    # 计算前一半和后一半的3mer频率
    result = []

    for seq in sequences:
        onepoint = int(len(seq) / 3)
        twopoint = int((len(seq)-onepoint)/2 + onepoint)
        # print(onepoint)
        # print(twopoint)
        first_half = seq[0:onepoint]
        mid_half = seq[onepoint:twopoint]
        second_half = seq[twopoint:len(seq)]
        first_half_freq = calculate_3mer_frequency(first_half)
        mid_half_freq = calculate_3mer_frequency(mid_half)

        second_half_freq = calculate_3mer_frequency(second_half)
        # print(first_half_freq)

        # 将前一半和后一半的3mer频率转换为字符串并拼接起来
        sequence_string1 = []
        for kmer in kmer_list:
            kmer_freq = first_half_freq.get(kmer, 0)/(len(first_half)-2)
            kmer_freq = kmer_freq
            sequence_string1.append(kmer_freq)
        # sequence_final1 = str(sequence_string1)
        sequence_string2 = []
        for kmer in kmer_list:
            kmer_freq = mid_half_freq.get(kmer, 0)/(len(mid_half)-2)
            kmer_freq = kmer_freq
            sequence_string2.append(kmer_freq)

        sequence_string3 = []
        for kmer in kmer_list:
            kmer_freq = second_half_freq.get(kmer, 0)/(len(second_half)-2)
            kmer_freq = kmer_freq
            sequence_string3.append(kmer_freq)
        # sequence_final2 = str(sequence_string)
        sequence_final = []
        sequence_final = sequence_string1 + sequence_string2 + sequence_string3
        ###编码后的结果加入列表
        result.append(sequence_final)
    # result = z_score_normalize(result)
    return result
# 生成128维的按照A，G，C，T顺序排列的3mer频率向量
sequence_vector = generate_3mer_vector(sequence)
sequence_array = np.array(sequence_vector)



# print(sequence_array)
df = pd.DataFrame(sequence_array.reshape(sequence_array.shape[0], -1))  # 将多维数组转为二维DataFrame
df.to_csv('./human/human_3mer_train.csv', index=True, header=False)  # 保存到CSV文件，不保存行索引和列头
