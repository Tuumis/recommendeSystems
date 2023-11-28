import numpy as np
import pandas as pd
import re
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 
from assignment3 import average_of_users_predictions, predictions_for_users, print_top_ten_recommendations

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

def handle_question(selected_users, pearson_correlation_data, ratings, question,):
    # Collect ratings of all 10 similar users of each group member 
    all_similar_users = []
    for user in selected_users:
        neighbors = search_nearest_neighbors(pearson_correlation_data,user,10)
        all_similar_users.extend(neighbors.index.values)
    ratings_of_neighbors = ratings.loc[all_similar_users]
    # Handle a why not question
    question = question.lower().split(' ')
    if 'genre' in question:
        index = question.index('genre')
        genre = question[index + 1]
        search_ratings_for_genre(ratings_of_neighbors,genre,)


# Searches ratings in some genre from given part of ratins
# Returns ratings of movies, which belong to given genre
def search_ratings_for_genre(ratings,genre):
    genre_movies = movies.loc[lambda m: m['genres'].str.lower().str.contains(genre.lower(), na=False)]['movieId']
    ratings_in_genre = ratings.loc[:, ratings.columns.isin(genre_movies)]
    return ratings_in_genre
    

# Searches from predictions, what is a first location of some genre in sorten movie recommendations list
# returns location (index starting from 0) + 1
# If a genre does not exist, returns -1
def search_location_of_genre(recommendations, genre):
    predictions = recommendations.sort_values(ascending=False)
    movieIds = predictions.index
    location = -1
    movie = []
    genres = []
    for i in range(0,len(movieIds)):
        movie = movieIds[i]
        genres = movies.query('movieId == @movie').get(key='genres').values[0]
        print(genres)
        if genre.lower() in genres.lower():
            location = i + 1
            break
    print(location)

def main():
    # Data to form where movieIds are columns and each row presents all ratings of one user
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    # Similarity of users with Pearson correlation
    ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

    selected_users = 249,353,456
    users_predictions = predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users)
    users_average_predictions = average_of_users_predictions(users_predictions)
    print_top_ten_recommendations(users_average_predictions['mean_rating'])
    # search_location_of_genre(users_average_predictions['mean_rating'],'animation')
    handle_question(selected_users, ratings_pearson_correlation, ratings_pivot, 'Why not genre animation in recommendations?')

if __name__ == "__main__":
    main()