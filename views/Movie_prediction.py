import streamlit as st
import pandas as pd
import joblib
import ast
from pathlib import Path


def show_prediction():

    st.title("🎬 Movie Success Prediction")

    # -----------------------------
    # Paths
    # -----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
    ENCODER_PATH = BASE_DIR / "models" / "genre_encoder.pkl"
    FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"
    DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"

    # -----------------------------
    # Load Files
    # -----------------------------
    try:
        model = joblib.load(MODEL_PATH)
        mlb = joblib.load(ENCODER_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        df = pd.read_csv(DATA_PATH)

    except Exception as e:
        st.error(f"Error loading files:\n{e}")
        return

    # -----------------------------
    # Extract Genres
    # -----------------------------
    def extract_genres(x):

        try:

            if pd.isna(x):
                return []

            # Handles list stored as string
            if x.startswith("["):

                genres = ast.literal_eval(x)

                if isinstance(genres, list):

                    if len(genres) > 0 and isinstance(genres[0], dict):
                        return [g["name"] for g in genres]

                    return genres

            # Handles comma separated genres
            return [g.strip() for g in x.split(",")]

        except:
            return []

    all_genres = sorted(
        set(
            genre
            for row in df["genres"].fillna("").apply(extract_genres)
            for genre in row
        )
    )

    # -----------------------------
    # User Inputs
    # -----------------------------
    st.subheader("Enter Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        budget = st.number_input(
            "Budget ($)",
            min_value=0.0,
            value=50000000.0,
            step=1000000.0
        )

        popularity = st.number_input(
            "Popularity",
            min_value=0.0,
            value=30.0
        )

    with col2:

        runtime = st.number_input(
            "Runtime (minutes)",
            min_value=30,
            max_value=300,
            value=120
        )

        vote_average = st.slider(
            "Vote Average",
            0.0,
            10.0,
            7.0,
            0.1
        )

    selected_genres = st.multiselect(
        "Select Genres",
        all_genres
    )

    st.divider()

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🎯 Predict Movie Success", use_container_width=True):

        genre_encoded = mlb.transform([selected_genres])[0]

        genre_dict = dict(zip(mlb.classes_, genre_encoded))

        input_data = {
            "budget": budget,
            "popularity": popularity,
            "runtime": runtime,
            "vote_average": vote_average,
        }

        input_data.update(genre_dict)

        input_df = pd.DataFrame([input_data])

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("✅ This movie is predicted to be SUCCESSFUL.")

        else:

            st.error("❌ This movie is predicted to be UNSUCCESSFUL.")

        st.metric(
            "Success Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

    st.divider()

    st.info(
        """
### 🤖 Model Information

This prediction is generated using a **Random Forest Classifier** trained on:

- Budget
- Popularity
- Runtime
- Vote Average
- Movie Genres

The model estimates the probability that a movie will be commercially successful.
"""
    )