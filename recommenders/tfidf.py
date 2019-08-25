import pandas as pd
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# get the data from: https://www.kaggle.com/tmdb/tmdb-movie-metadata
# load in the data
df = pd.read_csv('../large_files/tmdb_5000_movies.csv')


# convert the relevant data for each movie into a single string
# to be ingested by TfidfVectorizer
def genres_and_keywords_to_string(row):
  genres = json.loads(row['genres'])
  genres = ' '.join(''.join(j['name'].split()) for j in genres)

  keywords = json.loads(row['keywords'])
  keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
  return "%s %s" % (genres, keywords)


# create a new string representation of each movie
df['string'] = df.apply(genres_and_keywords_to_string, axis=1)


# create a tf-idf vectorizer object
# remove stopwords automatically
tfidf = TfidfVectorizer(max_features=2000)

# create a data matrix from the overviews
X = tfidf.fit_transform(df['string'])

# check the shape of X
print("X.shape:", X.shape)

# generate a mapping from movie title -> index (in df)
movie2idx = pd.Series(df.index, index=df['title'])

# create a function that generates recommendations
def recommend(title):
  # get the row in the dataframe for this movie
  idx = movie2idx[title]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]
  # print("idx:", idx)

  # calculate the pairwise similarities for this movie
  query = X[idx]
  scores = cosine_similarity(query, X)

  # currently the array is 1 x N, make it just a 1-D array
  scores = scores.flatten()

  # get the indexes of the highest scoring movies
  # get the first K recommendations
  # don't return itself!
  recommended_idx = (-scores).argsort()[1:6]

  # return the titles of the recommendations
  return df['title'].iloc[recommended_idx]


print("\nRecommendations for 'Scream 3':")
print(recommend('Scream 3'))

print("\nRecommendations for 'Mortal Kombat':")
print(recommend('Mortal Kombat'))

print("\nRecommendations for 'Runaway Bride':")
print(recommend('Runaway Bride'))
