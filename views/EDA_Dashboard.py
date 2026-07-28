import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


def show_eda():

    st.title("📊 Exploratory Data Analysis")

    # -----------------------------
    # Load Dataset
    # -----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"

    df = pd.read_csv(DATA_PATH)

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Filters")

    # Genre Filter
    genre_list = sorted(
        set(
            genre.strip()
            for genres in df["genres"]
            for genre in genres.split(",")
        )
    )

    selected_genres = st.sidebar.multiselect(
        "Select Genre",
        genre_list,
        default=genre_list
    )

    # Vote Average Filter
    min_vote = st.sidebar.slider(
        "Minimum Vote Average",
        float(df["vote_average"].min()),
        float(df["vote_average"].max()),
        float(df["vote_average"].min())
    )

    # Popularity Filter
    min_popularity = st.sidebar.slider(
        "Minimum Popularity",
        float(df["popularity"].min()),
        float(df["popularity"].max()),
        float(df["popularity"].min())
    )

    # -----------------------------
    # Apply Filters
    # -----------------------------
    filtered_df = df[
        (df["vote_average"] >= min_vote) &
        (df["popularity"] >= min_popularity)
    ]

    filtered_df = filtered_df[
        filtered_df["genres"].apply(
            lambda x: any(g.strip() in selected_genres for g in x.split(","))
        )
    ]

    # -----------------------------
    # KPI Cards
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Movies", len(filtered_df))

    col2.metric(
        "Average Budget",
        f"${filtered_df['budget'].mean()/1_000_000:.2f} M"
    )

    col3.metric(
        "Average Revenue",
        f"${filtered_df['revenue'].mean()/1_000_000:.2f} M"
    )

    st.markdown("---")

    # -----------------------------
    # Budget vs Revenue
    # -----------------------------
    st.subheader("💰 Budget vs Revenue")

    fig = px.scatter(
        filtered_df,
        x="budget",
        y="revenue",
        color="success",
        hover_name="title",
        title="Budget vs Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Genre Distribution
    # -----------------------------
    st.subheader("🎭 Genre Distribution")

    genre_df = filtered_df.copy()

    genre_df["genres"] = genre_df["genres"].str.split(",")

    genre_df = genre_df.explode("genres")

    genre_df["genres"] = genre_df["genres"].str.strip()

    genre_count = genre_df["genres"].value_counts().reset_index()

    genre_count.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_count,
        x="Genre",
        y="Count",
        title="Genre Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Genre Success Rate
    # -----------------------------
    st.subheader("🏆 Genre Success Rate")

    genre_success = (
        genre_df.groupby("genres")["success"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        genre_success,
        x="genres",
        y="success",
        title="Success Rate by Genre"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Popularity Distribution
    # -----------------------------
    st.subheader("⭐ Popularity Distribution")

    fig = px.histogram(
        filtered_df,
        x="popularity",
        nbins=30,
        title="Popularity Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Runtime Distribution
    # -----------------------------
    st.subheader("🎬 Runtime Distribution")

    fig = px.histogram(
        filtered_df,
        x="runtime",
        nbins=25,
        title="Runtime Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Vote Average Distribution
    # -----------------------------
    st.subheader("🌟 Vote Average Distribution")

    fig = px.histogram(
        filtered_df,
        x="vote_average",
        nbins=20,
        title="Vote Average Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    st.subheader("🔥 Correlation Heatmap")

    corr = filtered_df[
        [
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "success"
        ]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)