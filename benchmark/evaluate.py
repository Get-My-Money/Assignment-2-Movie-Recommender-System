# Import required libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

full_data = pd.read_csv("merged_data.csv")
movie_data = pd.read_csv("movie_info.csv")

data = full_data.copy()


# Assuming 'data' is your DataFrame
user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Transpose the user-item matrix
movie_user_matrix = user_item_matrix.T

# Create a mapping from movie_id to matrix row index
movie_to_index = {movie_id: i for i, movie_id in enumerate(movie_user_matrix.index)}

# Convert DataFrame of movie features to scipy sparse matrix
matrix_movie_user = csr_matrix(movie_user_matrix.values)

# Define model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

# Fit
model_knn.fit(matrix_movie_user)

def movie_recommender_engine(movie_id, matrix, cf_model, n_recs):
    # Calculate neighbour distances and indices
    distances, indices = cf_model.kneighbors(matrix[movie_to_index[movie_id]].reshape(1, -1), n_neighbors=n_recs+1)
        
    # Get list of indices for recommended movies
    rec_movie_indices = [i+1 for i in indices.flatten()][1:]  
        
    return rec_movie_indices



item1 = movie_data.copy()

# Calculate average ratings for 'gender_F'
female_avg_ratings = data[data['gender_F'] == 1].groupby('movie_id')['rating'].mean()
female_avg_ratings.name = 'Female_avg_rating'
item1 = item1.merge(female_avg_ratings, left_on='movie_id', right_index=True, how='left')

# Calculate average ratings for 'gender_M'
male_avg_ratings = data[data['gender_M'] == 1].groupby('movie_id')['rating'].mean()
male_avg_ratings.name = 'Male_avg_rating'
item1 = item1.merge(male_avg_ratings, left_on='movie_id', right_index=True, how='left')

bins = [7, 10, 14, 18, 25, 35, 45, 55, 65, np.inf]
labels = ['7-10', '11-14', '15-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)

# Calculate average ratings for each age group and merge with 'item1'
for label in labels:
    age_avg_ratings = data[data['age_group'] == label].groupby('movie_id')['rating'].mean()
    age_avg_ratings.name = f'{label}_avg_rating'
    item1 = item1.merge(age_avg_ratings, left_on='movie_id', right_index=True, how='left')
    
# Get the unique occupation types
occupation_types = data['occupation'].unique()

# Calculate average ratings for each occupation type and merge with 'item1'
for occupation in occupation_types:
    occupation_avg_ratings = data[data['occupation'] == occupation].groupby('movie_id')['rating'].mean()
    occupation_avg_ratings.name = f'{occupation}_avg_rating'
    item1 = item1.merge(occupation_avg_ratings, left_on='movie_id', right_index=True, how='left')

# Fill null values with 3
item1 = item1.fillna(3)



data1 = data.copy()
bins = [1922, 1950, 1970, 1980, 1990, np.inf]
labels = ['1922-1950', '1951-1970', '1971-1980', '1981-1990', '1991-1998']
data1['year_group'] = pd.cut(data1['release_year'], bins=bins, labels=labels)
data1['gender'] = np.where(data1['gender_F']==1, 'female', 'male')

genre_cols = ['unknown','Action', 'Adventure', 'Animation', 'Children',
          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
          'Sci-Fi', 'Thriller', 'War', 'Western']

# Initialize a DataFrame to store the average ratings
average_ratings = pd.DataFrame(columns=['user_id', 'gender', 'age', 'occupation'] + [f'{label}_avg_rating' for label in labels] + [f'{genre}_avg_rating' for genre in genre_cols])

# For each user
for user_id in data1['user_id'].unique():
    # Select the rows for this user
    user_rows = data1[data1['user_id'] == user_id]
    
    # Get the user's gender, age, and occupation
    gender = "male" if user_rows['gender_M'].iloc[0] == 1 else "female"
    age = user_rows['age'].iloc[0]
    occupation = user_rows['occupation'].iloc[0]
    
    # Calculate the average rating for each release period
    period_avg_ratings = [user_rows[user_rows['year_group'] == label]['rating'].mean() for label in labels]
    
    # Calculate the average rating for each genre
    genre_avg_ratings = [(user_rows[user_rows[genre] == 1]['rating'].mean()) for genre in genre_cols]
    
    # Append the average ratings to the DataFrame
    average_ratings = average_ratings.append(pd.Series([user_id, gender, age, occupation] + period_avg_ratings + genre_avg_ratings, index=average_ratings.columns), ignore_index=True)

# Fill null values with 3
average_ratings = average_ratings.fillna(3)


def calculate_film_rating(user_id, movie_id):
    # Get the user and movie data
    user_ratings = average_ratings[average_ratings['user_id'] == user_id]
    movie = item1[item1['movie_id'] == movie_id]

    # Calculate the genre part of the formula
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genre_ratings = [movie[genre].values[0] * user_ratings[f'{genre}_avg_rating'].values[0] for genre in genre_cols if movie[genre].values[0] == 1]
    genre_part = sum(genre_ratings) / len(genre_ratings) if genre_ratings else 0

    # Calculate gender rating
    gender_part = movie['Female_avg_rating'].values[0] if user_ratings['gender'].values[0] == "female" else movie['Male_avg_rating'].values[0]
    
    # Calculate age rating
    age = user_ratings["age"].values[0]
    if (age >= 7 and age <= 10):
        age_part = movie['7-10_avg_rating'].values[0]
    elif (age >= 11 and age <= 14):
        age_part = movie['11-14_avg_rating'].values[0]
    elif (age >= 15 and age <= 18):
        age_part = movie['15-18_avg_rating'].values[0]
    elif (age >= 19 and age <= 25):
        age_part = movie['19-25_avg_rating'].values[0]
    elif (age >= 26 and age <= 35):
        age_part = movie['26-35_avg_rating'].values[0]
    elif (age >= 36 and age <= 45):
        age_part = movie['36-45_avg_rating'].values[0]
    elif (age >= 46 and age <= 55):
        age_part = movie['46-55_avg_rating'].values[0]
    elif (age >= 56 and age <= 65):
        age_part = movie['56-65_avg_rating'].values[0]
    else:
        age_part = movie['65+_avg_rating'].values[0]
    
    # Calculate occupation rating
    occupation_part = movie[str(user_ratings["occupation"].values[0]) + "_avg_rating"].values[0]
    
    # Calculate release year rating
    try:
        year = int((movie["release_date"].values[0])[-4:])
    except:
        year = 1980
    if (year >= 1922 and year <= 1950):
        year_part = average_ratings['1922-1950_avg_rating'].values[0]
    elif (year >= 1951 and year <= 1970):
        year_part = average_ratings['1951-1970_avg_rating'].values[0]
    elif (year >= 1971 and year <= 1980):
        year_part = average_ratings['1971-1980_avg_rating'].values[0]
    elif (year >= 1981 and year <= 1990):
        year_part = average_ratings['1981-1990_avg_rating'].values[0]
    elif (year >= 1991 and year <= 1998):
        year_part = average_ratings['1991-1998_avg_rating'].values[0]

    # Calculate the final film rating
    film_rating = 0.3 * genre_part + 0.2 * gender_part + 0.2 * age_part + 0.2 * occupation_part + 0.1 * year_part

    return film_rating




recommendations = {}

# Loop through all movie_ids
for i in range(len(list(set(data["user_id"])))):
    # Get watched movies from our user
    watched_movies = list(data[data['user_id'] == i+1]["movie_id"])
    
    # Get the top rated movie for each user
    top_movies_for_user = list(data[data['user_id'] == i+1].nlargest(4, 'rating')["movie_id"])
    good_movies = list(set(movie_recommender_engine(top_movies_for_user[0], matrix_movie_user, model_knn, 5) + movie_recommender_engine(top_movies_for_user[1], matrix_movie_user, model_knn, 5) + movie_recommender_engine(top_movies_for_user[2], matrix_movie_user, model_knn, 5) + movie_recommender_engine(top_movies_for_user[3], matrix_movie_user, model_knn, 5)))
    
    # Get 5 random unwatched movies that is good for our user
    unwatched_good_movies = [movie for movie in good_movies if movie not in watched_movies]
    unwatched_good_movies = random.sample(unwatched_good_movies, min(len(unwatched_good_movies), 5))
    
    # Add some random films if we have less than 5 in unwatched_good_movies list
    if len(unwatched_good_movies) < 5:
        random_films = random.sample([film for film in list(set(data["movie_id"])) if film not in watched_movies], 5-len(unwatched_good_movies))
        unwatched_good_movies += random_films
    
    recommendations[i+1] = unwatched_good_movies



def goodness_of_recommendations(recommendations):
    # Caculate average recommendation rating for all our users
    counter = 0
    summator = 0

    for i in range(len(recommendations)):
        for movie_id in recommendations[i+1]:
            counter += 1
            summator += calculate_film_rating(i+1, movie_id)

    average_recommendation_rating = summator/counter
    
    
    # Calculate best and worst films that can be suggested
    best_ratings = []
    worst_ratings = []
    for index in recommendations:
        ratings = []
        for i in range(1, 1682):
            ratings.append(calculate_film_rating(index, i))
        ratings.sort()
        best_ratings.append(np.mean(ratings[-5:]))
        worst_ratings.append(np.mean(ratings[:5]))
    best_recommendation_rating = np.mean(best_ratings)
    worst_recommendation_rating = np.mean(worst_ratings)
    
    
    # Calculate and return our rating and metric
    return average_recommendation_rating, (average_recommendation_rating - worst_recommendation_rating)/(best_recommendation_rating - worst_recommendation_rating)

rating, goodness_score = goodness_of_recommendations(recommendations)

print("Average recommendation rating:", rating)
print("Recommendations goodness_score:", goodness_score)


