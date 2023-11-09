import numpy as np
import pandas as pd
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

def average_of_users_predictions(users_predictions):
    users_average_predictions = []
    for movie in range(0,users_predictions[0].size):
        sum_of_movie_rating = 0
        for user in users_predictions:
            sum_of_movie_rating += user[movie]
        users_average_predictions.append(sum_of_movie_rating/len(users_predictions))
    return users_average_predictions

def predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users):
    users_predictions = []
    for user in selected_users:
        neighbors = search_nearest_neighbors(ratings_pearson_correlation,user,10)
        ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
        selected_user_ratings = ratings_pivot.loc[user]
        prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)
        users_predictions.append(prediction_of_movies.values)
    return users_predictions

def misery_of_users_predictions(users_predictions):
    users_misery_predictions = []
    for movie in range(0,users_predictions[0].size):
        first_iteration = True
        for user in users_predictions:
            if np.isnan(user[movie]) == True:
                rating = user[movie]
                break
            elif first_iteration == True:
                rating = user[movie]
                first_iteration = False
            elif user[movie] < rating:
                rating = user[movie]
        users_misery_predictions.append(rating)
    return users_misery_predictions

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
    users_predictions = predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users)
    users_average_predictions = pd.Series(average_of_users_predictions(users_predictions))
    print('Average predictions:')
    print_top_ten_recommendations(users_average_predictions)

    users_misery_predictions = pd.Series(misery_of_users_predictions(users_predictions))
    print('Least misery predictions:')
    print_top_ten_recommendations(users_misery_predictions)

if __name__ == "__main__":
    main()