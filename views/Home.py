import streamlit as st
import pandas as pd
import ast
from pathlib import Path
import matplotlib.pyplot as plt

def show_home():

    st.title("🎬 MovieIQ")

    # ------------------------------------------------
    # Load Data
    # ------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"

    df = pd.read_csv(DATA_PATH)

    # ------------------------------------------------
    # Convert Genres
    # ------------------------------------------------

    def extract_genres(x):
        try:
            return [i["name"] for i in ast.literal_eval(x)]
        except:
            return []

    df["genres"] = df["genres"].fillna("[]").apply(extract_genres)

    # ------------------------------------------------
    # Header
    # ------------------------------------------------

    st.markdown("""
    ### Movie Analytics & Success Prediction

    Explore movie trends, analyze datasets, visualize insights,
    and predict movie success using Machine Learning.
    """)

    st.markdown("---")

    # ------------------------------------------------
    # KPI Cards
    # ------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🎬 Total Movies", len(df))
    col2.metric("⭐ Avg Rating", round(df["vote_average"].mean(), 2))
    col3.metric("🔥 Avg Popularity", round(df["popularity"].mean(), 2))
    col4.metric("💰 Avg Budget", f"${df['budget'].mean():,.0f}")

    st.markdown("---")

    # ------------------------------------------------
    # Budget Distribution
    # ------------------------------------------------

    st.subheader("💰 Budget Distribution")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df["budget"], bins=40)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Movie Budget Distribution")

    st.pyplot(fig)

    # ------------------------------------------------
    # Genre Distribution
    # ------------------------------------------------

    st.subheader("🎭 Genre Distribution")

    genre_series = df.explode("genres")

    genre_count = genre_series["genres"].value_counts()

    fig, ax = plt.subplots(figsize=(10,5))
    genre_count.plot(kind="bar", ax=ax)

    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.set_title("Genre Distribution")

    st.pyplot(fig)

    # ------------------------------------------------
    # Top Rated Movies
    # ------------------------------------------------

    st.subheader("⭐ Top Rated Movies")

    top_movies = (
        df.sort_values("vote_average", ascending=False)
        [["title", "vote_average", "popularity"]]
        .head(10)
    )

    st.dataframe(top_movies, use_container_width=True)

    # ------------------------------------------------
    # Highest Revenue Movies
    # ------------------------------------------------

    st.subheader("💰 Highest Revenue Movies")

    revenue_movies = (
        df.sort_values("revenue", ascending=False)
        [["title", "revenue"]]
        .head(10)
    )

    st.dataframe(revenue_movies, use_container_width=True)

    # ------------------------------------------------
    # Most Popular Movies
    # ------------------------------------------------

    st.subheader("🔥 Most Popular Movies")

    popular_movies = (
        df.sort_values("popularity", ascending=False)
        [["title", "popularity"]]
        .head(10)
    )

    st.dataframe(popular_movies, use_container_width=True)

    # ------------------------------------------------
    # Footer
    # ------------------------------------------------

    st.markdown("---")

    st.info("""
    MovieIQ is an end-to-end data analytics project built using:

    • Python
    • Pandas
    • Matplotlib
    • Streamlit
    • Scikit-Learn
    • Random Forest Classifier
    """)