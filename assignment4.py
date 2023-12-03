import numpy as np
import pandas as pd
import re
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 
from assignment3 import average_of_users_predictions, predictions_for_users, print_top_ten_recommendations

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

# Given an answer depending on the stats the item got from the reviews etc.
def answer_by_ratings(ratings, item, group_predictions, itemId):
    no_review = 0
    review = 0
    sum_of_ratings = 0
    liked = 0
    disliked = 0
    average = 0
    for rating in ratings:
        if np.isnan(rating):
            no_review += 1
        else:
            review += 1
            sum_of_ratings += rating
            if rating >= 3:
                liked += 1
            else:
                disliked += 1
    if (review != 0):
        average = sum_of_ratings/review
    if (no_review == ratings.size):
        print("None of your peers has rated", item)
    elif (liked < disliked):
        print(liked, "peers like", item, "but", disliked, "dislike it.")
    elif (average < 3):
        print("Average for", item, "by your peers are only", average)
    elif(group_predictions[itemId] < 3):
        print("Predicted score for", item, "is only", group_predictions[itemId])
    else:
        print(item, "got lower rating prediction then those that are listed.")
    print()

# Forms explanations for genre related questions 
# Aggregates results using ratings of similar users and movies in genre rated by them
def answer_by_ratings_genre(ratings, genre):
    no_review = []
    review = []
    liked = []
    disliked = []
    average = []
    for user, ratings in ratings.items():
        no_rev = ratings.isnull().sum()
        rev = ratings.notnull().sum()
        likes = len(ratings.loc[lambda r : r >= 2.5])
        dislikes = len(ratings.loc[lambda r : r <= 2.5])
        rev_avg = np.mean(ratings)
        review.append(rev)
        no_review.append(no_rev)
        liked.append(likes)
        disliked.append(dislikes)
        average.append(rev_avg)
    if sum(review) == 0:
        print("Any of group members or similar users haven't rated any movies in genre", genre, "\n")
    elif (np.mean(liked) < np.mean(disliked)):
        print("On average", np.round(np.mean(liked)), "likes movies in genre", genre, "but", np.round(np.mean(disliked)), "dislikes.\n")
    else:
        most_liked = np.argmax(liked)
        most_liked = average[most_liked]
        print("Mostly", max(liked), "peers likes the same movie in genre", genre, "with average", np.round(most_liked,1), "\n")

# Checks existance of the movie
def check_if_movie_exist(movie):
    title_to_check = re.sub(r'\(\d{4}\)', '', movie).strip()
    movies_with_index = movies.set_index('movieId')
    movie_titles = movies_with_index['title']
    matching_movies = movie_titles[movie_titles.str.contains(title_to_check, case=False)]
    if not matching_movies.empty:
        length_differences = matching_movies.apply(lambda x: abs(len(title_to_check) - len(x)))
        best_match_index = length_differences.idxmin()
        return best_match_index
    else:
        print(movie, "does not exist in database.\n") 
        return None

def handle_question(selected_users, pearson_correlation_data, ratings, group_predictions, question,k=10):
    print(question)
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
        genre_ratings = search_ratings_for_genre(ratings_of_neighbors,genre)
        # Check if dataset contains movies in that genre
        if len(genre_ratings.columns) < 1:
            print('Movies in this genre does not exist\n')
            return
        if search_location_of_genre(group_predictions,genre) < (2 * k):
            print(k, "is too small number of predictions\n")
            return
        answer_by_ratings_genre(genre_ratings,genre)
    else:
        if 'first?' in question:
            name = ' '.join(question[3:-1]).capitalize()
        else:
            name = ' '.join(question[2:])[:-1].capitalize()
        movie = check_if_movie_exist(name)
        if movie is None:
            return
        if search_location_of_movie(group_predictions,movie) < (2 * k):
            print(k, "is too small number of predictions\n")
            return
        ratings_for_movie = search_ratings_for_movie(ratings_of_neighbors, movie)
        answer_by_ratings(ratings_for_movie, name, group_predictions, movie)
        

# Searches ratings in some genre from given part of ratins
# Returns ratings of movies, which belong to given genre
def search_ratings_for_genre(ratings,genre):
    genre_movies = movies.loc[lambda m: m['genres'].str.lower().str.contains(genre.lower(), na=False)]['movieId']
    ratings_in_genre = ratings.loc[:, ratings.columns.isin(genre_movies)]
    ratings_in_genre_avg = np.mean(ratings_in_genre,axis=1)
    return ratings_in_genre
    
def search_ratings_for_movie(ratings, movie):
    return ratings[movie]


# Searches from predictions, what is a first location of some genre in sorten movie recommendations list
# returns location (index starting from 1)
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
        #print(genres)
        if genre.lower() in genres.lower():
            location = i + 1
            break
    return location

def search_location_of_movie(recommendations, movie):
    predictions = recommendations.sort_values(ascending=False)
    location = -1
    movieIds = predictions.index
    for i in range(0,len(movieIds)):
        location = i + 1
        if (movie == movieIds[i]):
            break
    return location

def main():
    # Data to form where movieIds are columns and each row presents all ratings of one user
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    # Similarity of users with Pearson correlation
    ratings_pearson_correlation = ratings_pivot.T.corr('pearson')

    selected_users = 249,353,456
    users_predictions = predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users)
    users_average_predictions = average_of_users_predictions(users_predictions)
    predictions_for_group =users_average_predictions['mean_rating']
    print("Top 10 reccomendations:")
    print_top_ten_recommendations(predictions_for_group)
    print("\n***Questions and answers***")
    handle_question(selected_users, ratings_pearson_correlation, ratings_pivot, predictions_for_group, 'Why not genre documentary in recommendations?')
    handle_question(selected_users, ratings_pearson_correlation, ratings_pivot, predictions_for_group, 'Why not Matrix?')
    handle_question(selected_users, ratings_pearson_correlation, ratings_pivot, predictions_for_group, 'Why not rank Matrix first?')

if __name__ == "__main__":
    main()