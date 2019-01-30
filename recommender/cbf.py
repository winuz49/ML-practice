#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math


ratings = pd.read_csv("./ratings.csv", encoding="ISO-8859-1")
movies = pd.read_csv("./movies.csv", encoding="ISO-8859-1")
tags = pd.read_csv("tags.csv", encoding="ISO-8859-1")
print 'ratings:'
print ratings.keys()
print ratings.describe()
print ratings.head()
print 'movies:'
print movies.keys()
print movies.describe()
print movies.head()
print 'tags:'
print tags.keys()
print tags.describe()
print tags.head()

'''
rate = ratings[ratings['userId'] == 320]
print rate
rate = rate[rate['movieId'] == 108]
print rate
'''

print tags[tags['movieId'] == 1197]['tag']


def calculate_tf_idf():
    # 计算TF-IDF
    tf = tags.groupby(['movieId', 'tag'], as_index=False, sort=False).count().rename(
        columns={'userId': 'tag_count_TF'})[['movieId', 'tag', 'tag_count_TF']]
    tf_distinct = tf[['tag', 'movieId']].drop_duplicates()

    df = tf_distinct.groupby(['tag'], as_index=False, sort=False).count().rename(
        columns={'movieId': 'tag_count_DF'})[['tag', 'tag_count_DF']]
    movie_len = len(np.unique(tags['movieId']))
    df['IDF'] = math.log10(movie_len) - np.log10(df['tag_count_DF'])
    tf = pd.merge(tf, df, on='tag', how='left', sort=False)
    tf['TF-IDF'] = tf['tag_count_TF'] * tf['IDF']

    print tf[['movieId', 'TF-IDF']]
    # 归一化
    vecs = tf[['movieId', 'TF-IDF']]
    vecs['TF-IDF-Sq'] = np.square(vecs['TF-IDF'])
    vecs = vecs.groupby(['movieId'], as_index=False, sort=False).sum().rename(
        columns={'TF-IDF-Sq': 'TF-IDF-Sq-sum'})[['movieId', 'TF-IDF-Sq-sum']]
    vecs['vect_len'] = np.sqrt(vecs[['TF-IDF-Sq-sum']].sum(axis=1))

    tf = pd.merge(tf, vecs, on='movieId', how='left', sort=False)
    tf['TAG_WT'] = tf['TF-IDF'] / tf['vect_len']

    print tf[['movieId', 'TAG_WT']]
    return tf


def calculate_up(tf):
    ratings_filter = ratings[ratings['rating'] >= 3.5]
    distinct_users = np.unique(ratings['userId'])
    user_tag_pref = pd.DataFrame()
    i = 1
    userID = 320
    user_index = distinct_users.tolist().index(userID)
    print user_index
    print len(distinct_users)

    for user in distinct_users:

        user_data = ratings_filter[ratings_filter['userId'] == user]
        user_data = pd.merge(tf, user_data, on='movieId', how='inner', sort=False)
        user_tmp = user_data.groupby(['tag'], as_index=False, sort=False).sum().rename(
            columns={'TAG_WT': 'tag_pref'})[['tag', 'tag_pref']]
        user_tmp['user'] = user
        user_tag_pref = user_tag_pref.append(user_tmp, ignore_index=True)

    return user_tag_pref


def recommend(userId):
    print 'recommend:'
    tag_merge_all = pd.DataFrame()

    tf = calculate_tf_idf()
    user_profile = calculate_up(tf)

    user_tag_pref = user_profile[user_profile['user'] == userId]
    distinct_movies = np.unique(tf['movieId'])
    for movie in distinct_movies:
        tf_movie = tf[tf['movieId'] == movie]
        tag_merge = pd.merge(tf_movie, user_tag_pref, on='tag', how='left', sort=False)
        tag_merge['tag_pref'] = tag_merge['tag_pref'].fillna(0)
        tag_merge['tag_value'] = tag_merge['TAG_WT'] * tag_merge['tag_pref']

        TAG_WT_val = np.sqrt(np.sum(np.square(tag_merge['TAG_WT']), axis=0))
        tag_pref_val = np.sqrt(np.sum(np.square(user_tag_pref['tag_pref']), axis=0))

        tag_merge_final = tag_merge.groupby(['user', 'movieId'])[['tag_value']].sum().rename(
            columns={'tag_value': 'Rating'}).reset_index()

        tag_merge_final['Rating'] = tag_merge_final['Rating'] / (TAG_WT_val * tag_pref_val)

        tag_merge_all = tag_merge_all.append(tag_merge_final, ignore_index=True)

    tag_merge_all = tag_merge_all.sort_values(['user', 'Rating'], ascending=False)
    movies_rated = ratings[ratings['userId'] == userId]['movieId']
    tag_merge_all = tag_merge_all[~tag_merge_all['movieId'].isin(movies_rated)]
    print tag_merge_all
    return tag_merge_all


if __name__ == "__main__":
    re_movies = recommend(320)
    movie_ids = re_movies['movieId']
    print type(movie_ids)

    for movie_id in movie_ids:
        print movies[movies['movieId'] == movie_id][['movieId', 'title']]

