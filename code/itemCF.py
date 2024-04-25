#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/4/20 16:36
# @Author : Lucas
# @File : itemCF.py
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# loadding the logger
logger = logging.getLogger("userCFLogger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("logger is loaded")

class ItemCF:
    def __init__(self, train_data_filepath, pretrain=False):
        """
        初始化ItemCF算法

        参数:
        ratings (numpy.array): 物品-用户评分矩阵，行为物品，列为用户
        """
        self.data_df = pd.DataFrame()
        self.item_user_matrix = np.zeros((0, 0))
        self.item_user_matrix = self._loadding_data_from_txt(train_data_filepath)
        self.n_items, self.n_users = self.item_user_matrix.shape
        self.similarity_matrix = self._cosine_similarity(pretrain=pretrain)
        pass

    def _loadding_data_from_txt(self, train_data_filepath):
        with open(train_data_filepath, 'r') as f:
            train_data = f.readlines()
            train_data = [line.strip().split('\t') for line in train_data]
            self.data_df = pd.DataFrame(train_data, columns=['user_id', 'item_id', 'rating'])
            self.data_df  = self.data_df.astype(int)
            # creeate user-item matrix
            self.item_user_matrix = self.data_df.pivot_table(index='item_id', columns='user_id', values='rating', fill_value=0)
            logger.info("train data is loaded")
            return self.item_user_matrix

    def _cosine_similarity(self, pretrain=False):
        """
        计算物品之间的余弦相似度矩阵
        """
        # 初始化相似度矩阵
        if pretrain:
            return np.loadtxt("item_similarity_matrix.txt")
        sim_matrix = cosine_similarity(self.item_user_matrix)
        # save the similarity matrix
        np.savetxt("item_similarity_matrix.txt", sim_matrix)
        logger.info("item similarity matrix is calculated and successfully save in item_similarity_matrix.txt")
        return sim_matrix

    def recommend_items(self, user_id, k=5):
        # 找出用户已评分的物品
        rated_items =  self.item_user_matrix.iloc[:, user_id].to_numpy().nonzero()[0]

        # 初始化推荐物品评分字典
        scores = {}

        for item in rated_items:
            for similar_item in np.argsort(self.similarity_matrix[item])[::-1][:k]:
                if similar_item not in rated_items and  self.item_user_matrix.iloc[similar_item, user_id] == 0:
                    if similar_item in scores:
                        scores[similar_item] += self.similarity_matrix[item][similar_item]
                    else:
                        scores[similar_item] = self.similarity_matrix[item][similar_item]

        # 对推荐物品按评分排序
        recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [item for item, _ in recommended_items[:k]]

if __name__ == '__main__':
    # 示例用法
    train_data_filename = 'train.txt'
    test_data_filename = 'test.txt'
    data_dir = '../data'
    save_dir = '../exp'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data_filepath = os.path.join(data_dir, train_data_filename)
    k = 5
    # 初始化ItemCF算法
    item_cf = ItemCF(train_data_filepath, pretrain=False)
    # 为用户推荐物品
    user_id = 2
    recommended_items = item_cf.recommend_items(user_id, k=k)
    save_path = os.path.join(save_dir, 'itemcf_recommend_items.txt')
    with open(save_path, 'w') as f:
        for user in range(0, item_cf.n_users):
            recommended_items = item_cf.recommend_items(user, k=10)
            for recommended_item in recommended_items:
                f.write(f"{user}\t{recommended_item}\n")
            logger.info(f"为用户{user}推荐的物品列表：{recommended_items}")
