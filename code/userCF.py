#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/4/18 11:42
# @Author : Lucas
# @File : userCF.py
# @Software: PyCharm

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

# loadding the train data and transfer to dataframe
'''
    train_df存储为三列表格，第一列是用户的编号，第二列是电影的编号，第三列是用户对已观看过的电影的评分
'''
class userCFClass:
    def __init__(self, train_data_filepath):
        self.data_df = pd.DataFrame()
        self.user_item_matrix = np.zeros((0, 0))
        self.user_similarity_matrix = np.zeros((0, 0))
        self.user_item_matrix = self._loadding_data_from_txt(train_data_filepath)
        self.n_users, self.n_users = self.user_item_matrix.shape
        self.user_similarity_matrix = self._calculate_user_similarity_matrix()
        pass

    def _loadding_data_from_txt(self, train_data_filepath):
        with open(train_data_filepath, 'r') as f:
            train_data = f.readlines()
            train_data = [line.strip().split('\t') for line in train_data]
            self.data_df = pd.DataFrame(train_data, columns=['user_id', 'item_id', 'rating'])
            self.data_df  = self.data_df .astype(float)
            # creeate user-item matrix
            self.user_item_matrix = self.data_df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
            logger.info("train data is loaded")
            return self.user_item_matrix

    def _calculate_user_similarity_matrix(self):
        # calculate the user similarity matrix by mse method
        self.user_similarity_matrix = self.user_item_matrix.T.corr().to_numpy()
        # self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        logger.info("user similarity matrix is calculated")
        return self.user_similarity_matrix

    # user-CF method
    def predict_ratings(self, user_id, item_id):
        # 计算目标用户与其他用户的相似度
        user_similarity = self.user_similarity_matrix[user_id - 1]
        # 找出相似度最高的k个用户
        similar_users_indices = np.argsort(user_similarity)[::-1][:k]
        # 计算预测评分
        weighted_sum = 0.0
        similarity_sum = 0.0
        for index in similar_users_indices:
            similar_user_id = index + 1
            if similar_user_id == user_id:
                continue
            # 获取相似用户对该物品的评分
            rating = self.data_df[(self.data_df['user_id'] == similar_user_id) & (self.data_df['item_id'] == item_id)]['rating'].values
            if len(rating) > 0:
                similarity = user_similarity[index]
                weighted_sum += rating * similarity
                similarity_sum += similarity
        if similarity_sum == 0.0:
            return 0.0
        else:
            predicted_rating = weighted_sum / similarity_sum
            return predicted_rating[0]

    def recommend(self, user_id, topk=5):
        # recommend the top k items for each user
        rating = self.user_item_matrix[self.user_item_matrix.index == user_id]
        # find the items which the user does not watch
        not_watch_items = rating.columns[rating.values[0] == 0].tolist()
        rating_df = pd.DataFrame(columns=['item_id', 'rating'])
        for item_id in not_watch_items:
            rating = self.predict_ratings(user_id, item_id)
            rating_df = rating_df.append({'item_id': item_id, 'rating': rating}, ignore_index=True)
        rating_df = rating_df.sort_values(by='rating', ascending=False)
        return rating_df['item_id'].values[:topk].astype(int)
if __name__ == '__main__':
    # variables
    train_data_filename = 'train.txt'
    test_data_filename = 'test.txt'
    data_dir = '../data'
    save_dir = '../exp'
    train_data_filepath = os.path.join(data_dir, train_data_filename)
    user_id = 0
    item_id = 1
    k = 5
    userCF = userCFClass(train_data_filepath=train_data_filepath)
    # predict the rating
    # predicted_rating = userCF.predict_ratings(user_id, item_id)
    # logger.info("predicted rating: {}".format(predicted_rating))
    # find the item which the user does not watch
    # recommend = userCF.recommend(user_id, topk=5)
    # logger.info("user {} recommend top {} items: {}".format(user_id, k, recommend))
    save_path = os.path.join(save_dir, 'usercf_recommend_items.txt')
    with open(save_path, 'w') as f:
        for user in range(0, userCF.n_users):
            recommended_items = userCF.recommend(user, topk=10)
            for recommended_item in recommended_items:
                f.write(f"{user}\t{recommended_item}\n")
            logger.info(f"为用户{user}推荐的物品列表：{recommended_items}")