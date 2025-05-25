from IPython import get_ipython
from IPython.display import display

!pip install pandas scikit-learn nltk google-generativeai

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
import random
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

genai.configure(api_key="AIzaSyBhbisT973pHc6h4xrHQnOSktaHHTOPNcA")
model = genai.GenerativeModel("models/gemini-1.5-flash")

movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, names=['movieId', 'title', 'genres'])
movies.to_csv('movies.csv', index=False)

movies = pd.read_csv('movies.csv')
movies['genres'] = movies['genres'].replace('(no genres listed)', '')
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movie(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return None
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

def parse_input(user_input):
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)
    for word in tokens:
        possible_matches = movies[movies['title'].str.lower().str.contains(word, na=False)]
        if not possible_matches.empty:
            return possible_matches.iloc[0]['title']
    return None

def ask_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "I'm having trouble getting insights right now."

def chatbot():
    greetings = ["Hello! I'm your movie buddy.", "Hi there! Ready to discover some movies?", "Hey! Let's find you a movie to watch."]
    bye = ["Goodbye!", "See you later!", "Happy watching!"]
    print(random.choice(greetings))
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'bye', 'quit']:
            print(random.choice(bye))
            break
        
        movie_title = parse_input(user_input)
        if movie_title:
            recommendations = recommend_movie(movie_title)
            if recommendations:
                print(f"\nBecause you mentioned '{movie_title}', you might also enjoy:")
                for rec in recommendations:
                    print("- " + rec)
                
                gemini_input = f"Tell me why someone who liked '{movie_title}' would enjoy these: {', '.join(recommendations)}."
                summary = ask_gemini(gemini_input)
                print("\nGemini says:")
                print(summary)
            else:
                print("Hmm... I couldn't find recommendations for that movie.")
        else:
            print("I'm not sure which movie you're talking about. Try mentioning a movie name!")

chatbot()
