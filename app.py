import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    return pd.read_csv('movies.csv')

def preprocess_data(data):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        data[feature] = data[feature].fillna('')
    data['combined_features'] = data['genres'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast'] + ' ' + data['director']
    return data

def compute_similarity(data):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(data['combined_features'])
    similarity = cosine_similarity(feature_vectors)
    return similarity

def recommend_movies(movie_name, data, similarity):
    list_of_all_titles = data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if len(find_close_match) == 0:
        st.error("Movie not found! Please try another one.")
        return
    close_match = find_close_match[0]
    index_of_the_movie = data[data.title == close_match].index.values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = data[data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    return recommended_movies

def main():
    st.title("Movie Recommendation App")
    movies_data = load_data()
    movies_data = preprocess_data(movies_data)
    similarity = compute_similarity(movies_data)

    movie_name = st.text_input('Enter your favorite movie name:')
    if st.button('Recommend'):
        recommended_movies = recommend_movies(movie_name, movies_data, similarity)
        if recommended_movies:
            st.success('Movies suggested for you:')
            for i, movie in enumerate(recommended_movies[:30], 1):
                st.write(f"{i}. {movie}")
        else:
            st.error('No recommendations available.')

if __name__ == '__main__':
    main()
