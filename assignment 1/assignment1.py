import numpy as np
import pandas as pd

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))
# all = pd.merge(movies, ratings)
print(ratings.head(3))
print(ratings.count())


# Data to form where movieIds are columns and each row presents all ratings of one user
ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

def search_nearest_neighbors(similarity_data, user_id, amount_of_neighbors):
    neighbors = similarity_data[user_id].sort_values(ascending=False).iloc[2:amount_of_neighbors+2]
    return (neighbors)

def search_ratings_of_neighbors(rating_data,neighbor_data):
    nearest_ratings = rating_data.loc[neighbor_data.index]
    return nearest_ratings

selected_user = 249
neighbors = search_nearest_neighbors(ratings_pearson_correlation,selected_user,10)
print(search_ratings_of_neighbors(ratings_pivot, neighbors))