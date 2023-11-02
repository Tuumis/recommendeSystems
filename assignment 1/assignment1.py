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

def predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors):
    user_similarity = neighbors.values
    ratings_values = ratings_of_neighbors.values
    mean_of_neighbors = ratings_of_neighbors.mean(axis=1, skipna=True).values
    mean_of_user = selected_user_ratings.mean(skipna=True)
    prediction_for_user = selected_user_ratings.copy()
    
    for movie in range(0,prediction_for_user.values.size):
        if np.isnan(prediction_for_user.values[movie]) == True:
            movie_score_compined = 0
            similarity_compined = 0
            for user in range(0,user_similarity.size):
                if np.isnan(ratings_values[user][movie]) == False:
                    movie_score_compined += user_similarity[user]*(ratings_values[user][movie]-mean_of_neighbors[user])
                    similarity_compined += user_similarity[user]
            if similarity_compined != 0:
                prediction_of_movie = mean_of_user + movie_score_compined/similarity_compined
                prediction_for_user.values[movie] = prediction_of_movie
            else: prediction_for_user.values[movie] = None
        else: prediction_for_user.values[movie] = None

    return prediction_for_user

selected_user = 249
neighbors = search_nearest_neighbors(ratings_pearson_correlation,selected_user,10)
ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
#print(ratings_of_neighbors)

selected_user_ratings = ratings_pivot.loc[selected_user]
print(selected_user_ratings)
prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)
print(prediction_of_movies)