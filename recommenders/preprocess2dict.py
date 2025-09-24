# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
#from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# load in the data
# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('.\\large_files\\movielens-20m-dataset\\small_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# a dictionary to tell us which users have rated which movies
user2movie = df_train.groupby('userId').movie_idx.agg(list).to_dict()
# a dicationary to tell us which movies have been rated by which users
movie2user = df_train.groupby('movie_idx').userId.agg(list).to_dict()
# a dictionary to look up ratings
user_movie_keys = zip(df_train.userId, df_train.movie_idx)
usermovie2rating = pd.Series(df_train.rating.values, index=user_movie_keys).to_dict()

# print("Calling: update_user2movie_and_movie2user")
# count = 0
# def update_user2movie_and_movie2user(row):
#   global count
#   count += 1
#   if count % 100000 == 0:
#     print("processed: %.3f" % (float(count)/cutoff))

#   i = int(row.userId)
#   j = int(row.movie_idx)
#   if i not in user2movie:
#     user2movie[i] = [j]
#   else:
#     user2movie[i].append(j)

#   if j not in movie2user:
#     movie2user[j] = [i]
#   else:
#     movie2user[j].append(i)

#   usermovie2rating[(i,j)] = row.rating
#df_train.apply(update_user2movie_and_movie2user, axis=1)

# test ratings dictionary
user_movie_keys_test = zip(df_test.userId, df_test.movie_idx)
usermovie2rating_test = pd.Series(df_test.rating.values, index=user_movie_keys_test).to_dict()

# print("Calling: update_usermovie2rating_test")
# count = 0
# def update_usermovie2rating_test(row):
#   global count
#   count += 1
#   if count % 100000 == 0:
#     print("processed: %.3f" % (float(count)/len(df_test)))

#   i = int(row.userId)
#   j = int(row.movie_idx)
#   usermovie2rating_test[(i,j)] = row.rating
# df_test.apply(update_usermovie2rating_test, axis=1)

# note: these are not really JSONs
with open('.\\large_files\\movielens-20m-dataset\\user2movie.json', 'wb') as f:
  pickle.dump(user2movie, f)

with open('.\\large_files\\movielens-20m-dataset\\movie2user.json', 'wb') as f:
  pickle.dump(movie2user, f)

with open('.\\large_files\\movielens-20m-dataset\\usermovie2rating.json', 'wb') as f:
  pickle.dump(usermovie2rating, f)

with open('.\\large_files\\movielens-20m-dataset\\usermovie2rating_test.json', 'wb') as f:
  pickle.dump(usermovie2rating_test, f)
