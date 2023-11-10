def predictions_for_users(ratings_pivot, ratings_pearson_correlation,selected_users):
    # users_predictions = []
    movieIds = ratings_pivot.columns
    users_predictions = pd.DataFrame(columns=movieIds)
    for user in selected_users:
        neighbors = search_nearest_neighbors(ratings_pearson_correlation,user,10)
        ratings_of_neighbors = search_ratings_of_neighbors(ratings_pivot, neighbors)
        selected_user_ratings = ratings_pivot.loc[user]
        prediction_of_movies = predict_movie_score(selected_user_ratings, neighbors, ratings_of_neighbors)
        users_predictions.loc[len(users_predictions)] = prediction_of_movies.values
        # users_predictions.append(prediction_of_movies.values)
    return users_predictions
