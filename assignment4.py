import numpy as np
import pandas as pd
import re
from assignment1 import search_nearest_neighbors, search_ratings_of_neighbors, predict_movie_score 
from assignment3 import average_of_users_predictions, predictions_for_users, print_top_ten_recommendations

# Reading data from files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv",usecols=range(3))

def answer_by_ratings(ratings, item):
    #print(ratings)
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
    #Tulostukset voi siivota pois kun vaan niitten avulla saa järkevät tulokset hommaan.
    print("no_review", no_review)
    print("reviw", review)
    if (review != 0):
        average = sum_of_ratings/review
        print("average", sum_of_ratings/review)
    print("liked", liked)
    print("disliked", disliked)
    print(ratings.size)
    if (no_review == ratings.size):
        print("None of your peers has rated", item)
    elif (liked < disliked):
        print(liked, "peers like", item, "but", disliked, "dislike it")
    else:
        print("Only", liked, "peers liked", item)

def check_if_movie_exist(movie):
    title_to_check = re.sub(r'\(\d{4}\)', '', movie).strip()
    movies_with_index = movies.set_index('movieId')
    movie_titles = movies_with_index['title']
    matching_movies = movie_titles[movie_titles.str.contains(title_to_check, case=False)]
    if not matching_movies.empty:
        length_differences = matching_movies.apply(lambda x: abs(len(title_to_check) - len(x)))
        best_match_index = length_differences.idxmin()
        #Seuraavat kaksi mahdollisesti kommentoitua riviä voi poistaa kun testailu valmis
        #best_match_title = movies_with_index.loc[best_match_index, 'title']
        #print(best_match_index, best_match_title)
        return best_match_index
    else:
        print(movie, "does not exist in database.") 
        return None

def handle_question(selected_users, pearson_correlation_data, ratings, question,):
    # Collect ratings of all 10 similar users of each group member 
    print(question)
    all_similar_users = []
    for user in selected_users:
        neighbors = search_nearest_neighbors(pearson_correlation_data,user,10)
        all_similar_users.extend(neighbors.index.values)
    ratings_of_neighbors = ratings.loc[all_similar_users]
    #print(ratings_of_neighbors)
    # Handle a why not question
    question = question.lower().split(' ')
    if 'genre' in question:
        index = question.index('genre')
        genre = question[index + 1]
        search_ratings_for_genre(ratings_of_neighbors,genre,)
    elif 'rank' in question:
        '''movie = check_if_movie_exist(question[3])
        if movie is None:
            return'''
        
    else:
        name = ' '.join(question[2:])[:-1].capitalize()
        print(name)
        movie = check_if_movie_exist(name)
        if movie is None:
            return
        ratings_for_movie = search_ratings_for_movie(ratings_of_neighbors, movie)
        answer_by_ratings(ratings_for_movie, name)
        



# Searches ratings in some genre from given part of ratins
# Returns ratings of movies, which belong to given genre
def search_ratings_for_genre(ratings,genre):
    genre_movies = movies.loc[lambda m: m['genres'].str.lower().str.contains(genre.lower(), na=False)]['movieId']
    ratings_in_genre = ratings.loc[:, ratings.columns.isin(genre_movies)]
    return ratings_in_genre
    
def search_ratings_for_movie(ratings, movie):
    return ratings[movie]

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
    handle_question(selected_users, ratings_pearson_correlation, ratings_pivot, 'Why not Matrix?')
    handle_question(selected_users, ratings_pearson_correlation, ratings_pivot, 'Why not rank Matrix first?')

if __name__ == "__main__":
    main()