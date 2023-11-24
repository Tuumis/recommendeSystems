import numpy as np
import pandas as pd
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

def average_of_users_predictions(users_predictions):
    movie_means = users_predictions.mean(axis=0, skipna=True)
    users_average_predictions = pd.DataFrame({'mean_rating': movie_means})
    users_average_predictions = users_average_predictions.dropna(subset=['mean_rating'])
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

def print_top_ten_recommendations(recomendations):
    highest_predictions = recomendations.sort_values(ascending=False).head(10)
    # MovieIds of top 10 predictions 
    highest_predictions_ids = highest_predictions.index

    # Printing movie titles of 10 highest predictions for selected user
    for prediction in highest_predictions_ids:
        movie = movies.query('movieId == @prediction')
        print(movie.get(key='title').values)

def weighted_average_of_users_predictions(users_predictions, weights):
    users_predictions = users_predictions.dropna(axis=1, how='all')
    weighted_predictions = users_predictions * np.array(weights)[:, np.newaxis]
    weighted_sum = np.sum(weighted_predictions, axis=0)
    sum_of_weights = np.sum(weights)
    users_weighted_average_predictions = pd.DataFrame({'weighted_mean_rating': weighted_sum / sum_of_weights})
    return users_weighted_average_predictions

def weights_withuser_satisfaction(users_predictions, group_predictions):
    weights = []
    group_predictions_top_10 = group_predictions.sort_values('mean_rating',ascending=False).head(10)
    # group_predictions_sorted = (group_predictions_sorted - group_predictions_sorted.min()) /(group_predictions_sorted.max() - group_predictions_sorted.min())
    print(group_predictions_top_10)
    for index, user in users_predictions.iterrows():
        user_predictions_for_group_top_10 = user.loc[group_predictions_top_10.index]
        average_of_individual_predictions = user.mean()
        predicted_avg_or_higher = user_predictions_for_group_top_10.loc[lambda r : r > average_of_individual_predictions]
        print(predicted_avg_or_higher)
        weight = 1 - (len(predicted_avg_or_higher) / 10)
        weights.append(weight)
    return weights    

def main():
    # Data to form where movieIds are columns and each row presents all ratings of one user
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    # Similarity of users with Pearson correlation
    ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

    selected_users = 249,353,456
    users_predictions = predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users)
    # print(users_predictions)
    users_average_predictions = average_of_users_predictions(users_predictions)
    # print_top_ten_recommendations(users_average_predictions['mean_rating'])
    weights = weights_withuser_satisfaction(users_predictions, users_average_predictions)
    print(users_average_predictions)
    
    for i in range(0,3):
        weighted_average_predictions = weighted_average_of_users_predictions(users_predictions, weights)
        print(weighted_average_predictions)
        weights = weights_withuser_satisfaction(weighted_average_predictions, users_average_predictions)
    print(weighted_average_predictions)
    #print_top_ten_recommendations(weighted_average_predictions)
        
if __name__ == "__main__":
    main()

# Weighted average (Thomas)
# normalizing and distance calculation (Minja)
# Show results for selected users 
# slides and argumentation 


