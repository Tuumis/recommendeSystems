import numpy as np
import pandas as pd
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 
from assignment3 import average_of_users_predictions, predictions_for_users, print_top_ten_recommendations

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

def main():
    # Data to form where movieIds are columns and each row presents all ratings of one user
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    # Similarity of users with Pearson correlation
    ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

    selected_users = 249,353,456
    users_predictions = predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users)
    users_average_predictions = average_of_users_predictions(users_predictions)
    print_top_ten_recommendations(users_average_predictions['mean_rating'])

if __name__ == "__main__":
    main()