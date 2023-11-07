import numpy as np
import pandas as pd
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 

def average_of_users_predictions(ratings_pivot, ratings_pearson_correlation,selected_users):
    for user in selected_users:
        neighbors = search_nearest_neighbors(ratings_pearson_correlation,user,10)
        ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
        selected_user_ratings = ratings_pivot.loc[user]
        prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)
        print(prediction_of_movies)

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