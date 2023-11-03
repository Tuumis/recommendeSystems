import numpy as np
import pandas as pd

# (a)
# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

# Data to form where movieIds are columns and each row presents all ratings of one user
ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')

# (d)
# Function gets user similarity data for example pearson correlation 
# and user whose neighbours wanted and wanted amount of neighbors
# It returns a dataframe containing wanted amount of most similar users
def search_nearest_neighbors(similarity_data, user_id, amount_of_neighbors):
    neighbors = similarity_data[user_id].sort_values(ascending=False).iloc[1:amount_of_neighbors+1]
    return (neighbors)

# (d)
# Returns ratings of neighbors given as a parameter.
# Searches ratings from ratings dataframe by userIds of neighbors
def search_ratings_of_neighbors(rating_data,neighbor_data):
    nearest_ratings = rating_data.loc[neighbor_data.index]
    return nearest_ratings

# (c)
# Predics movie ratings for selected user by using neighbors ratings.
# Prediction is made with an common prediction function.
# If neighbor have not rated the movie then we skip that neighbor from the function
# Eventually returns a series that contains None values for movies user have already rated and
# None values for movies none of the neighbors have reviewed and for the others the rating.
def predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors):
    user_similarity = neighbors.values
    ratings_values = ratings_of_neighbors.values
    mean_of_neighbors = ratings_of_neighbors.mean(axis=1, skipna=True).values
    mean_of_user = selected_user_ratings.mean(skipna=True)
    # Copys series of users ratings to be used for predicted ratings
    prediction_for_user = selected_user_ratings.copy()
    
    # Looping through all the movies and skipping and setting None value for those user has reviewed
    for movie in range(0,prediction_for_user.values.size):
        if np.isnan(prediction_for_user.values[movie]) == True:
            movie_score_compined = 0
            similarity_compined = 0
            # Looping through all the neighbors, skippings those who have not reviewed the movie
            # If none of the neighbors have reviewed the movie we set prediction for movie None
            # In Loop we calculate their ratings by comparing is it higher or lover than their average and
            # use similarity to weight the rating.Then we calculate them together. In the end we divide the result
            # with mean of neighbors similarity and add it to mean of selected users to get prediction of the movie.
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

# (a)
print(ratings.head(3))
print(ratings.count())

# (b)
# Similarity of users with Pearson correlation
ratings_pearson_correlation = ratings_pivot.T.corr('pearson')
print(ratings_pearson_correlation)

# (d)
# Selecting user and searching 10 nearest users
selected_user = 249
neighbors = search_nearest_neighbors(ratings_pearson_correlation,selected_user,10)
ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
print(ratings_of_neighbors)

# (d)
# Ratings of selected user
selected_user_ratings = ratings_pivot.loc[selected_user]
print(selected_user_ratings)

# (b/d)
# Counting predictions to all movies that are rated any of neighbors
prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)

# (d)
highest_predictions = prediction_of_movies.sort_values(ascending=False).head(10)
# MovieIds of top 10 predictions 
highest_predictions_ids = highest_predictions.index

# (d)
# Printing movie titles of 10 highest predictions for selected user
for prediction in highest_predictions_ids:
    movie = movies.query('movieId == @prediction')
    print(movie.get(key='title').values)

# (e)
# Spearman rank correlation works beter if it the data is limited and the movie
# distribution might not follow normal distribution. Spearman rank correlation
# handles efficenly the non-linear patterns instead of Pearson correlation.
# Could be added that one good thing is that the bouth are implemented same way
# which allows efficent usage debending the wanted results or the dataset in case.
# Implementation happens the same way as Pearson correlation:
ratings_spearman_correlation = ratings_pivot.T.corr('spearman')