#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/4/25 8:55
# @Author : Lucas
# @File : evaluate.py
import logging
import os

data_dir = '../data'
exp_dir = '../exp'
test_filename = 'test.txt'
predictions_filename = 'itemcf_recommend_items.txt'
test_path = os.path.join(data_dir, test_filename)
predictions_path = os.path.join(exp_dir, predictions_filename)

# loadding the logger
logger = logging.getLogger("evaluateLogger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("logger is loaded")

# evaluate the recommendation results
import pandas as pd

def calculate_tp_fp_fn(predictions_path, test_path):
    # Load the recommended movies and the actual liked movies
    recommended_movies = pd.read_csv(predictions_path, sep='\t', header=None, names=['user', 'item'])
    actual_liked_movies = pd.read_csv(test_path, sep='\t', header=None, names=['user', 'item', 'rating'])
    # Group the data by user
    recommended_movies_grouped = recommended_movies.groupby('user')['item'].apply(set).to_dict()
    actual_liked_movies_grouped = actual_liked_movies.groupby('user')['item'].apply(set).to_dict()

    tp = 0
    fp = 0
    fn = 0

    # Calculate TP, FP, and FN for each user
    for user, recommended in recommended_movies_grouped.items():
        actual = actual_liked_movies_grouped.get(user, set())
        tp += len(recommended & actual)  # Intersection is TP
        fp += len(recommended - actual)  # In recommended but not in actual is FP
        fn += len(actual - recommended)  # In actual but not in recommended is FN

    return tp, fp, fn

tp, fp, fn = calculate_tp_fp_fn(predictions_path, test_path)
print(f"TP: {tp}, FP: {fp}, FN: {fn}")

def calculate_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1_score

precision, recall, f1_score = calculate_metrics(tp, fp, fn)
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1 Score: {f1_score:.4f}")