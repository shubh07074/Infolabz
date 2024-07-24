import pickle
import streamlit as st
import requests
import pandas as pd
import base64

st.set_page_config(
    page_title="MovieMatcher",
    page_icon="icon.png"
)

def fetch_poster_and_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=6fcfe36d2ac8608dede699f5f505abea&language=en-US"
    data = requests.get(url).json()
    if 'poster_path' in data:
        poster_path = data['poster_path']
        full_path = f"https://image.tmdb.org/t/p/original/{poster_path}"
        tmdb_url = f"https://www.themoviedb.org/movie/{movie_id}"
        return full_path, tmdb_url
    else:
        return "https://via.placeholder.com/500x750?text=No+Image", "#"

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_urls = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        poster, tmdb_url = fetch_poster_and_url(movie_id)
        recommended_movie_posters.append(poster)
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_urls.append(tmdb_url)

    return recommended_movie_names, recommended_movie_posters, recommended_movie_urls

# Function to encode image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to the background image
background_image_path = "back.jpg"

# Encode the image
base64_image = get_base64_of_image(background_image_path)

# Inject CSS to apply the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/jpeg;base64,{base64_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.header('Movie Recommender System')
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies)

movie_list = movies['title'].tolist()
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_urls = recommend(selected_movie)
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"[![{recommended_movie_names[idx]}]({recommended_movie_posters[idx]})]({recommended_movie_urls[idx]})")
            # Display the title only once in a larger font
            st.markdown(f"<span style='font-size: 16px; word-wrap: break-word;'>{recommended_movie_names[idx]}</span>", unsafe_allow_html=True)
