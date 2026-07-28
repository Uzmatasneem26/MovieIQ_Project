import streamlit as st
import pandas as pd
import plotly.express as px
import ast
from pathlib import Path
import streamlit as st

def show_home():

    st.title("🎬 MovieIQ")

#------------------------------------------------
# Page Config
# ------------------------------------------------

    st.set_page_config(
        page_title="MovieIQ Dashboard",
        layout="wide"
    )

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

    st.title("🎬 MovieIQ Dashboard")

    st.markdown(
"""
### Movie Analytics & Success Prediction

Explore movie trends, analyze datasets, visualize insights,
and predict movie success using Machine Learning.
"""
)

    st.markdown("---")

# ------------------------------------------------
# KPI Cards
# ------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🎬 Total Movies",
        len(df)
        )

    col2.metric(
        "⭐ Avg Rating",
        round(df["vote_average"].mean(),2)
        )

    col3.metric(
        "🔥 Avg Popularity",
        round(df["popularity"].mean(),2)
        )

    col4.metric(
        "💰 Avg Budget",
        f"${df['budget'].mean():,.0f}"
        )
    st.markdown("---")

# ------------------------------------------------
# Budget Distribution
# ------------------------------------------------
    st.subheader("💰 Budget Distribution")

    fig = px.histogram(
        df,
        x="budget",
        nbins=40,
        title="Movie Budget Distribution"
        )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Genre Distribution
# ------------------------------------------------

    genre_series = df.explode("genres")

    genre_count = genre_series["genres"].value_counts().reset_index()

    genre_count.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_count,
        x="Genre",
        y="Count",
        title="Genre Distribution"
        )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Top Rated Movies
# ------------------------------------------------

    st.subheader("⭐ Top Rated Movies")

    top_movies = df.sort_values(
        "vote_average",
        ascending=False
    ) [["title","vote_average","popularity"]].head(10)

    st.dataframe(
        top_movies,
        use_container_width=True
        )

# ------------------------------------------------
# Highest Revenue
# ------------------------------------------------

    st.subheader("💰 Highest Revenue Movies")

    revenue_movies = df.sort_values(
        "revenue",
        ascending=False
        )[["title","revenue"]].head(10)

    st.dataframe(
        revenue_movies,
        use_container_width=True
        )

# ------------------------------------------------
# Most Popular Movies
# ------------------------------------------------

    st.subheader("🔥 Most Popular Movies")

    popular_movies = df.sort_values(
        "popularity",
        ascending=False
        )[["title","popularity"]].head(10)

    st.dataframe(
        popular_movies,
        use_container_width=True
      )

# ------------------------------------------------
# Footer
# ------------------------------------------------

    st.markdown("---")

    st.info(
"""
MovieIQ is an end-to-end data analytics project built using:

- Python
- Pandas
- Plotly
- Streamlit
- Scikit-Learn
- Random Forest Classifier
"""
    )