import numpy as np
import pandas as pd
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

# a) i)
# Counts average rating for movie by users predictions
def average_of_users_predictions(users_predictions):
    movie_means = users_predictions.mean(axis=0, skipna=True)
    users_average_predictions = pd.DataFrame({'mean_rating': movie_means})
    users_average_predictions = users_average_predictions.dropna(subset=['mean_rating'])
    print(users_average_predictions)
    return users_average_predictions

# Counts predictions to multiple users by using funktions from assigment1.py 
def predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users):
    movieIds = ratings_pivot.columns
    users_predictions = pd.DataFrame(columns=movieIds)
    for user in selected_users:
        neighbors = search_nearest_neighbors(ratings_pearson_correlation,user,10)
        ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
        selected_user_ratings = ratings_pivot.loc[user]
        prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)
        users_predictions.loc[len(users_predictions)] = prediction_of_movies.values
    return users_predictions

# a) ii)
# Gets misery prediction among the users
def misery_of_users_predictions(users_predictions):
    movie_min_ratings = users_predictions.min(axis=0, skipna=True)
    users_misery_predictions = pd.DataFrame({'misery_rating': movie_min_ratings})
    users_misery_predictions = users_misery_predictions.dropna(subset=['misery_rating'])
    return users_misery_predictions

# b)
# Counts misery average distance: 
# distance of min rating and average and the distance is subtracted from average rating
def misery_avg_distance(users_predictions):
    movie_min_ratings = users_predictions.min(axis=0, skipna=True)
    movie_means = users_predictions.mean(axis=0, skipna=True)
    differences = np.abs(movie_means - movie_min_ratings)
    predictions = movie_means - differences
    predictions = pd.DataFrame({'misery_avg_predictions': predictions})
    predictions = predictions.dropna(subset=['misery_avg_predictions'])
    return predictions

# For printing the recomendations
def print_top_ten_recommendations(recomendations):
    highest_predictions = recomendations.sort_values(ascending=False).head(10)
    # MovieIds of top 10 predictions 
    highest_predictions_ids = highest_predictions.index

    # Printing movie titles of 10 highest predictions for selected user
    for prediction in highest_predictions_ids:
        movie = movies.query('movieId == @prediction')
        print(movie.get(key='title').values)


def main():
    # Data to form where movieIds are columns and each row presents all ratings of one user
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    # Similarity of users with Pearson correlation
    ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

    selected_users = 249,353,456
    # a) i)
    users_predictions = predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users)
    users_average_predictions = average_of_users_predictions(users_predictions)
    print('Average predictions:')
    print_top_ten_recommendations(users_average_predictions['mean_rating'])

    # a) ii)
    users_misery_predictions = misery_of_users_predictions(users_predictions)
    print('\nLeast misery predictions:')
    print_top_ten_recommendations(users_misery_predictions['misery_rating'])
    misery_average_distance_predictions = misery_avg_distance(users_predictions)

    # b)
    print('\nMisery average distance predictions')
    print_top_ten_recommendations(misery_average_distance_predictions['misery_avg_predictions'])
        
if __name__ == "__main__":
    main()