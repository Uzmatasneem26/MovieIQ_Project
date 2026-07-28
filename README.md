# 🎬 MovieIQ - Movie Success Prediction System

MovieIQ is an end-to-end Machine Learning project that predicts whether a movie is likely to be successful based on its characteristics. The project also provides an interactive dashboard for Exploratory Data Analysis (EDA), Statistical Testing, Machine Learning Model Evaluation, and Movie Success Prediction using Streamlit.

---

# 📌 Project Overview

Movie success depends on several factors such as budget, popularity, runtime, ratings, and genre.

This project analyzes historical movie data, performs statistical analysis, builds a Machine Learning model, and provides an interactive Streamlit application for prediction and visualization.

---

# 🚀 Features

✅ Interactive Home Dashboard

✅ Exploratory Data Analysis (EDA)

- Budget vs Revenue
- Genre Distribution
- Genre Success Rate
- Popularity Distribution
- Runtime Distribution
- Vote Average Distribution
- Correlation Heatmap

✅ Statistical Tests

- Independent Samples T-Test
- Chi-Square Test

✅ Movie Success Prediction

- User input form
- Random Forest prediction
- Success probability

✅ Machine Learning Dashboard

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Feature Importance
- Classification Report

---

# 📂 Project Structure

```
MovieIQ_Project/
│
├── app.py
│
├── data/
│   └── movies_cleaned.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── genre_encoder.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│
├── assets/
│
├── views/
│   ├── __init__.py
│   ├── 1_Home.py
│   ├── 2_EDA_Dashboard.py
│   ├── 3_Statistical_Tests.py
│   ├── 4_Movie_prediction.py
│   ├── 5_ML_Model_Dashboard.py
│   └── 6_About.py
│
├── data_cleaning.py
├── train_model.py
├── Requirements.txt
└── README.md
```

---

# 📊 Dataset

The dataset contains information about movies including:

- Title
- Budget
- Revenue
- Popularity
- Runtime
- Vote Average
- Genres
- Success Label

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- Handled missing values
- Feature selection
- Genre extraction
- MultiLabelBinarizer encoding
- Success label creation
- Feature engineering

---

# 📈 Exploratory Data Analysis

The dashboard includes:

- Budget vs Revenue Scatter Plot
- Genre Distribution
- Genre Success Rate
- Popularity Histogram
- Runtime Histogram
- Vote Average Histogram
- Correlation Heatmap

---

# 📊 Statistical Analysis

### Independent Samples T-Test

Objective:

Determine whether popularity differs significantly between successful and unsuccessful movies.

### Chi-Square Test

Objective:

Determine whether movie genre is associated with movie success.

---

# 🤖 Machine Learning

Algorithm Used

- Random Forest Classifier

Features Used

- Budget
- Popularity
- Runtime
- Vote Average
- Genres

Target Variable

- Success

Train-Test Split

- 80 : 20

---

# 📉 Model Evaluation

The dashboard displays:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Feature Importance
- Classification Report

---

# 🎯 Movie Prediction

Users can enter:

- Budget
- Popularity
- Runtime
- Vote Average
- Genres

The application predicts:

- Movie Success
- Success Probability

---

# 🛠 Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

### Machine Learning

- Scikit-learn
- SciPy

### Web Application

- Streamlit

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/MovieIQ_Project.git
```

Move into the project folder

```bash
cd MovieIQ_Project
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r Requirements.txt
```

---

# ▶ Running the Application

Run

```bash
streamlit run app.py
```

The application will open automatically at

```
http://localhost:8501
```

---

# 📷 Application Pages

🏠 Home

📊 EDA Dashboard

📈 Statistical Tests

🎯 Movie Prediction

🤖 ML Model Dashboard

ℹ About

---

# 🎯 Future Improvements

- Movie Recommendation System
- Deep Learning Models
- Hyperparameter Tuning
- Model Comparison Dashboard
- TMDB API Integration
- Deployment on Streamlit Cloud

---

# 👩‍💻 Developed By

**Uzma Tasneem R**

Data Analyst | Python | SQL | Machine Learning | Streamlit | Power BI

---

# ⭐ If you found this project useful, consider giving it a star.