# Movie-Recommender
This repository contains code for building a movie recommendation system based on text features extracted from movie metadata. The system utilizes natural language processing techniques to analyze movie attributes such as genres, keywords, taglines, cast, and directors to recommend similar movies.

## Features

- **Data Loading**: The `movies.csv` file is loaded into the system, containing movie metadata such as titles, genres, keywords, taglines, cast, and directors.
  
- **Data Preprocessing**: Missing values in selected features (genres, keywords, taglines, cast, and director) are filled with empty strings to facilitate further processing.
  
- **Feature Engineering**: Text features from different columns (genres, keywords, taglines, cast, and director) are combined into a single feature called `combined_features`.
  
- **Vectorization**: The combined text features are transformed into numerical representations using TF-IDF vectorization.
  
- **Similarity Calculation**: Cosine similarity is computed between the feature vectors of movies to measure their similarity.

- **Recommendation Generation**: Given a movie title input, the system identifies the most similar movies and recommends them to the user.

## Instructions

To run the movie recommendation system:

1. Clone this repository to your local machine.
   
2. Run the Streamlit app using the following command:
   streamlit run app.py

3. Input your favorite movie title and click on the "Recommend" button to get personalized movie recommendations.

## Data Source

The movie metadata used in this project is sourced from the `movies.csv` file.

