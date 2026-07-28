import streamlit as st

from views.Home import show_home
from views.EDA_Dashboard import show_eda
from views.Statistical_Tests import show_statistics
from views.Movie_prediction import show_prediction
from views.ML_Model_Dashboard import show_ml_dashboard
from views.About import show_about

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# ---------------- Sidebar ----------------
with st.sidebar:

    st.markdown("# 🎬 MovieIQ")
    st.markdown("### Navigation")

    page = st.radio(
        "",
        (
            "🏠 Home",
            "📊 EDA Dashboard",
            "📈 Statistical Tests",
            "🎯 Movie Prediction",
            "🤖 ML Model Dashboard",
            "ℹ️ About"
        )
    )

# ---------------- Routing ----------------
if page == "🏠 Home":
    show_home()

elif page == "📊 EDA Dashboard":
    show_eda()

elif page == "📈 Statistical Tests":
    show_statistics()

elif page == "🎯 Movie Prediction":
    show_prediction()

elif page == "🤖 ML Model Dashboard":
    show_ml_dashboard()

elif page == "ℹ️ About":
    show_about()