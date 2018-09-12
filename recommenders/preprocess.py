# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pandas as pd

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('../large_files/movielens-20m-dataset/rating.csv')



# note:
# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('../large_files/movielens-20m-dataset/edited_rating.csv', index=False)