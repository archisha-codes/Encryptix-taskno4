import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load the dataset
movies = pd.read_csv('C:/Users/MADHU PRIYA/Desktop/New folder/aiml tasks/movies_metadata.csv', low_memory=False)

# Display the first few rows of the dataset
print(movies.head())

# Function to extract genres from the JSON-like string
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'].lower() for genre in genres]
    except (ValueError, SyntaxError):
        return []

# Apply the function to the genres column
movies['genres'] = movies['genres'].apply(extract_genres)

# Display the processed genres
print(movies[['title', 'genres']].head())

# Create a genre list to encode genres as binary vectors
genre_list = set(genre for sublist in movies['genres'] for genre in sublist)

# Create a binary matrix for genres
for genre in genre_list:
    movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

# Drop the original genres column
movies = movies.drop('genres', axis=1)

# Keep the 'title' column aside for later use
titles = movies['title']

# Convert non-numeric columns to numeric if possible, otherwise drop them
non_numeric_columns = movies.select_dtypes(exclude=['number']).columns
movies = movies.drop(non_numeric_columns, axis=1, errors='ignore')

# Handle missing values by filling them with zero
movies = movies.fillna(0)

# Display the processed dataset with binary genre matrix
print(movies.head())

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(movies)

# Display the cosine similarity matrix
print(cosine_sim)

# Function to recommend movies
def recommend_movies(title, num_recommendations=5):
    # Find the index of the movie in the dataset
    idx = titles[titles.str.lower() == title.lower()].index[0]
    
    # Get similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top recommended movies
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the recommended movie titles
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = titles.iloc[movie_indices]
    
    return recommended_movies

# Example usage
recommendations = recommend_movies('The Matrix')
print(recommendations)
