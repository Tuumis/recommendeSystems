import numpy as np
import pandas as pd
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 

def average_of_users_predictions(ratings_pivot, ratings_pearson_correlation,selected_users):
    users_predictions = []
    for user in selected_users:
        neighbors = search_nearest_neighbors(ratings_pearson_correlation,user,10)
        ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
        selected_user_ratings = ratings_pivot.loc[user]
        prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)
        users_predictions.append(prediction_of_movies.values)
    
    users_average_predictions = []
    for movie in range(0,users_predictions[0].size):
        sum_of_movie_rating = 0
        for user in users_predictions:
            sum_of_movie_rating += user[movie]
        users_average_predictions.append(sum_of_movie_rating/len(users_predictions))
    return users_average_predictions



def main():
    # Reading data from files
    movies = pd.read_csv("ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

    # Data to form where movieIds are columns and each row presents all ratings of one user
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    # Similarity of users with Pearson correlation
    ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

    selected_users = 249,353,456
    average_of_users_predictions(ratings_pivot, ratings_pearson_correlation,selected_users)
    

if __name__ == "__main__":
    main()