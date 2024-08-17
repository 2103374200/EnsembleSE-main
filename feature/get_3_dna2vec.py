import os
import sys
from multiprocessing import Process
import csv
import Bio
import gensim
import numpy as np
import pandas as pd
from Bio import SeqIO

NULL_vec=np.zeros((64))     ###设置一个全0数组

def get_kmer(dnaSeq,K):
    dnaSeq=dnaSeq.upper()
    l=len(dnaSeq)
    return [dnaSeq[i:i+K] for i in range(0,l-K+1,K)]
def part_vec(sequence,K):
    kmers = get_kmer(sequence, K)
    code = []
    for kmer in kmers:
        if ('n' not in kmer) and ('N' not in kmer):
            code.append(embedding_matrix[kmer])
        else:
            code.append(NULL_vec)
    array = np.array(code)
    ave = array.sum(axis=0)
    return ave
def seq_to_vec(cell_name,sequence,embedding_matrix,K):
    code_file = './merge/{}{}mer-3datavec.csv'.format(cell_name, K)
    seqid=1
    for seq in sequence:
        onepoint = int(len(seq) / 3)
        twopoint = int((len(seq) - onepoint) / 2 + onepoint)
        first_half = seq[0:onepoint]
        mid_half = seq[onepoint:twopoint]
        second_half = seq[twopoint:len(seq)]
        seq_first= part_vec(first_half,K)
        seq_mid = part_vec(mid_half, K)
        seq_second = part_vec(second_half, K)
        ave = np.concatenate((seq_first, seq_mid,seq_second))
        ave = pd.DataFrame(ave).T
        id = pd.DataFrame([seqid]).T
        ave=pd.concat([id,ave],axis=1,ignore_index=True)
        ave = ave.reset_index(drop=True)
        ave.to_csv(code_file,index=False,mode='a',header=False)
        seqid+=1


if __name__=='__main__':

    cell_name = 'test_merge'
    seq_file = '../data/datasets/test_merge.csv'
    with open(seq_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        p_sequence = []
        n_sequence = []
        for row in csv_reader:
            if 'N' not in row[0]:
                if row[1] == '0':
                    n_sequence.append(row[0])
                if row[1] == '1':
                    p_sequence.append(row[0])
            else:
                if row[1] == '0':
                    row[0] = row[0].replace('N', '')
                    n_sequence.append(row[0])
                if row[1] == '1':
                    row[0] = row[0].replace('N', '')
                    p_sequence.append(row[0])
        sequence = []
        sequence = p_sequence + n_sequence
        print(len(p_sequence))
        print(len(n_sequence))
        print(len(sequence))
    embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format("h_dna2vec-64.w2v")
    records_num=len(sequence)
    ps=[]
    for K in range(3,7):    
        p=Process(target=seq_to_vec,args=(cell_name,sequence,embedding_matrix,K))
        ps.append(p)
    for i in range(4):
        ps[i].start()
    for i in range(4):
        ps[i].join()
    print('The main process %s is done...' % os.getpid())
