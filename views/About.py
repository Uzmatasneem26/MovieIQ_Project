import streamlit as st


def show_about():

    st.title("ℹ️ About MovieIQ")

    st.markdown("""
    ## 🎬 Project Overview

    **MovieIQ** is a Machine Learning and Data Analytics project that predicts whether a movie is likely to be successful using historical movie data.

    The project combines **Exploratory Data Analysis (EDA), Statistical Testing, Machine Learning, and Interactive Visualizations** to help understand the factors influencing movie success.

    ---
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Dataset Features")

        st.markdown("""
        The prediction model uses:

        - 💰 Budget
        - ⭐ Popularity
        - 🎬 Runtime
        - 🌟 Vote Average
        - 🎭 Genres
        """)

    with col2:

        st.subheader("🤖 Machine Learning")

        st.markdown("""
        **Algorithm**

        - Random Forest Classifier

        **Target Variable**

        - Movie Success (Success / Failure)
        """)

    st.divider()

    st.header("🛠️ Technologies Used")

    tech1, tech2 = st.columns(2)

    with tech1:

        st.markdown("""
        ### Programming

        - Python
        - Pandas
        - NumPy
        - SciPy
        """)

    with tech2:

        st.markdown("""
        ### Visualization & ML

        - Streamlit
        - Plotly
        - Matplotlib
        - Scikit-learn
        """)

    st.divider()

    st.header("📈 Model Evaluation")

    st.markdown("""
    The model performance is evaluated using:

    - ✅ Accuracy
    - ✅ Precision
    - ✅ Recall
    - ✅ F1 Score
    - ✅ ROC Curve
    - ✅ Confusion Matrix
    - ✅ Feature Importance
    """)

    st.divider()

    st.header("✨ Application Features")

    st.markdown("""
    ✔ Home Dashboard

    ✔ Exploratory Data Analysis

    ✔ Statistical Hypothesis Testing

    ✔ Movie Success Prediction

    ✔ Machine Learning Performance Dashboard

    ✔ About Project
    """)

    st.divider()

    st.header("👩‍💻 Developed By")

    st.success("""
    **Uzma Tasneem R**

    Data Analyst | Python | SQL | Power BI | Machine Learning

    Portfolio Project – MovieIQ
    """)