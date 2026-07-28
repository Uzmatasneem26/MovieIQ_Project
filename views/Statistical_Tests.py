import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from pathlib import Path


def show_statistics():

    # -----------------------------------------------------
    # Page Title
    # -----------------------------------------------------
    st.title("📊 Statistical Tests")
    st.markdown("This page performs hypothesis testing on the MovieIQ dataset.")

    # -----------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error("❌ movies_cleaned.csv not found inside the data folder.")
        return

    # -----------------------------------------------------
    # Remove Duplicate Columns
    # -----------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()]

    # -----------------------------------------------------
    # Check Required Columns
    # -----------------------------------------------------
    required_columns = ["success", "popularity", "genres"]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        return

    # =====================================================
    # Independent Samples T-Test
    # =====================================================

    st.header("1️⃣ Independent Samples T-Test")

    st.markdown("""
    **Objective**

    Determine whether the popularity differs significantly between successful and unsuccessful movies.

    **Hypotheses**

    - **H₀:** There is no significant difference in popularity.
    - **H₁:** There is a significant difference in popularity.
    """)

    successful = df[df["success"] == 1]["popularity"].dropna()
    unsuccessful = df[df["success"] == 0]["popularity"].dropna()

    t_stat, p_value = ttest_ind(
        successful,
        unsuccessful,
        equal_var=False
    )

    alpha = 0.05

    col1, col2 = st.columns(2)

    with col1:
        st.metric("T-Statistic", f"{t_stat:.4f}")

    with col2:
        st.metric("P-Value", f"{p_value:.6f}")

    if p_value < alpha:
        st.success(
            "Reject H₀\n\nPopularity differs significantly between successful and unsuccessful movies."
        )
    else:
        st.info(
            "Fail to Reject H₀\n\nNo significant difference in popularity."
        )

    st.divider()

    # =====================================================
    # Chi-Square Test
    # =====================================================

    st.header("2️⃣ Chi-Square Test")

    st.markdown("""
    **Objective**

    Determine whether movie genre is associated with movie success.

    **Hypotheses**

    - **H₀:** Genre and success are independent.
    - **H₁:** Genre and success are associated.
    """)

    genre_df = df.copy()

    genre_df["genres"] = genre_df["genres"].fillna("Unknown")
    genre_df["genres"] = genre_df["genres"].astype(str)
    genre_df["genres"] = genre_df["genres"].str.split(",")

    genre_df = genre_df.explode("genres", ignore_index=True)

    genre_df["genres"] = genre_df["genres"].str.strip()

    genre_df = genre_df[
        genre_df["genres"].notna() &
        (genre_df["genres"] != "")
    ]

    contingency = pd.crosstab(
        genre_df["genres"],
        genre_df["success"]
    )

    st.subheader("Contingency Table")
    st.dataframe(contingency, use_container_width=True)

    chi2, p, dof, expected = chi2_contingency(contingency)

    c1, c2, c3 = st.columns(3)

    c1.metric("Chi-Square", f"{chi2:.4f}")
    c2.metric("P-Value", f"{p:.6f}")
    c3.metric("Degrees of Freedom", dof)

    if p < alpha:
        st.success(
            "Reject H₀\n\nGenre is significantly associated with movie success."
        )
    else:
        st.info(
            "Fail to Reject H₀\n\nNo significant association between genre and success."
        )

    st.divider()

    # =====================================================
    # Interpretation
    # =====================================================

    st.header("📌 Interpretation")

    st.info("""
    **Decision Rule**

    • P-value < 0.05 → Reject the Null Hypothesis

    • P-value ≥ 0.05 → Fail to Reject the Null Hypothesis

    **Significance Level (α) = 0.05**
    """)

    st.divider()

    # =====================================================
    # Dataset Preview
    # =====================================================

    with st.expander("📂 Preview Dataset"):
        st.dataframe(df.head(), use_container_width=True)