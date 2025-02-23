Movie Recommendation System

This project implements a movie recommendation system based on TF-IDF and cosine similarity. Users input a description of a movie they like, and the system finds the most similar movies based on their plots.

Dataset:

The dataset consists of movie plots collected from Wikipedia and IMDb.

Structure:
Column Name	Description
rank	Movie ranking 
title	Movie title
genre	Movie genres
wiki_plot	Plot summary from Wikipedia
imdb_plot	Plot summary from IMDb

Check "data" folder for entire CSV file. The dataset is automatically loaded when running the script.

Setup:

Ensure you have Python 3.8+ installed. 

Virtual Environment Setup:

Mac/Linux:
python3 -m venv env
source env/bin/activate

Windows:
python -m venv env
env\Scripts\activate

Install necessary dependencies: pip install -r requirements.txt

Running the Code:

python3 movie_recommendation.py

Enter your movie description: (ENTER MOVIE DESCRIPTION HERE)

Enter number of matches to return: (ENTER NUMBER OF MATCHES TO RETURN HERE)


