from email.mime import application
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bs4 import BeautifulSoup
import pickle
import requests
import os
from functools import lru_cache

# Global variables to store loaded data
data = None
similarity = None
clf = None
vectorizer = None

def load_models():
    global clf, vectorizer
    try:
        if not os.path.exists('nlp_model.pkl') or not os.path.exists('tranform.pkl'):
            raise FileNotFoundError("Required model files are missing. Please run train_models.py first.")
        
        # Load the pre-trained model and vectorizer
        with open('nlp_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('tranform.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

@lru_cache(maxsize=1)
def create_similarity():
    global data, similarity
    try:
        if not os.path.exists('main_data.csv'):
            raise FileNotFoundError("main_data.csv is missing")
            
        data = pd.read_csv('main_data.csv')
        # Create a CountVectorizer for movie recommendations
        cv = CountVectorizer(max_features=5000, stop_words='english')
        count_matrix = cv.fit_transform(data['comb'])
        similarity = cosine_similarity(count_matrix)
        return data, similarity
    except Exception as e:
        print(f"Error creating similarity matrix: {str(e)}")
        raise

def rcmd(m):
    global data, similarity
    m = m.lower()
    try:
        if data is None or similarity is None:
            data, similarity = create_similarity()
            
        if m not in data['movie_title'].unique():
            return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
            
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x:x[1], reverse=True)
        lst = lst[1:11]  # excluding first item since it is the requested movie itself
        return [data['movie_title'][a] for a, _ in lst]
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return "An error occurred while generating recommendations"

# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    try:
        # getting data from AJAX request
        title = request.form['title']
        cast_ids = request.form['cast_ids']
        cast_names = request.form['cast_names']
        cast_chars = request.form['cast_chars']
        cast_bdays = request.form['cast_bdays']
        cast_bios = request.form['cast_bios']
        cast_places = request.form['cast_places']
        cast_profiles = request.form['cast_profiles']
        imdb_id = request.form['imdb_id']
        poster = request.form['poster']
        genres = request.form['genres']
        overview = request.form['overview']
        vote_average = request.form['rating']
        vote_count = request.form['vote_count']
        release_date = request.form['release_date']
        runtime = request.form['runtime']
        status = request.form['status']
        rec_movies = request.form['rec_movies']
        rec_posters = request.form['rec_posters']

        # get movie suggestions for auto complete
        suggestions = get_suggestions()

        # call the convert_to_list function for every string that needs to be converted to list
        rec_movies = convert_to_list(rec_movies)
        rec_posters = convert_to_list(rec_posters)
        cast_names = convert_to_list(cast_names)
        cast_chars = convert_to_list(cast_chars)
        cast_profiles = convert_to_list(cast_profiles)
        cast_bdays = convert_to_list(cast_bdays)
        cast_bios = convert_to_list(cast_bios)
        cast_places = convert_to_list(cast_places)
        
        # convert string to list (eg. "[1,2,3]" to [1,2,3])
        cast_ids = cast_ids.strip('[]').split(',')
        
        # rendering the string to python string
        for i in range(len(cast_bios)):
            cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
        
        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
        casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
        cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

        # web scraping to get user reviews from IMDB site
        url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})

            reviews_list = []
            reviews_status = []
            
            for reviews in soup_result:
                if reviews.string:
                    reviews_list.append(reviews.string)
                    movie_review_list = np.array([reviews.string])
                    # Transform the review using the vectorizer
                    movie_vector = vectorizer.transform(movie_review_list)
                    pred = clf.predict(movie_vector)
                    reviews_status.append('Good' if pred else 'Bad')

            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

            return render_template('recommend.html',
                                title=title, poster=poster, overview=overview,
                                vote_average=vote_average, vote_count=vote_count,
                                release_date=release_date, runtime=runtime,
                                status=status, genres=genres, movie_cards=movie_cards,
                                reviews=movie_reviews, casts=casts, cast_details=cast_details)
                                
        except requests.RequestException as e:
            print(f"Error fetching IMDB reviews: {str(e)}")
            return render_template('recommend.html',
                                title=title, poster=poster, overview=overview,
                                vote_average=vote_average, vote_count=vote_count,
                                release_date=release_date, runtime=runtime,
                                status=status, genres=genres, movie_cards=movie_cards,
                                reviews={}, casts=casts, cast_details=cast_details)
                                
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == '__main__':
    try:
        load_models()
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
