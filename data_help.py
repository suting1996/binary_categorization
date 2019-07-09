import numpy as np
import re

def load_data_and_labels(postive_file, negative_file):
    """
    positive:---> [1,0]
    negative:--->[0,1]
    (X,Y)
    """
    positive_train = list(open(postive_file, "r",encoding='utf8').readlines())
    positive_train = [s.strip() for s in positive_train]
    negative_train = list(open(negative_file, "r",encoding='utf8').readlines())
    negative_train = [s.strip() for s in negative_train]
    x_text = positive_train + negative_train
    x_text = [sent for sent in x_text]
    # 定义类别标签 ，格式为one-hot的形式: y=positive--->[1,0]
    positive_labels = [[1, 0] for _ in positive_train]
    # print positive_labels[1:3]
    negative_labels = [[0, 1] for _ in negative_train]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    利用迭代器从训练数据的返回某一个batch的数据
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # 每回合打乱顺序
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]