import streamlit as st
import pandas as pd
import joblib
import ast
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)
from sklearn.model_selection import train_test_split


def show_ml_dashboard():

    st.title("🤖 Machine Learning Dashboard")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"
    MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
    ENCODER_PATH = BASE_DIR / "models" / "genre_encoder.pkl"
    FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"

    # --------------------------------------------------------
    # Load Files
    # --------------------------------------------------------

    try:

        df = pd.read_csv(DATA_PATH)

        model = joblib.load(MODEL_PATH)

        mlb = joblib.load(ENCODER_PATH)

        feature_columns = joblib.load(FEATURES_PATH)

    except Exception as e:

        st.error(f"Error loading files:\n{e}")

        return

    # --------------------------------------------------------
    # Extract Genres
    # --------------------------------------------------------

    def extract_genres(x):

        try:

            if pd.isna(x):
                return []

            if x.startswith("["):

                genres = ast.literal_eval(x)

                if isinstance(genres, list):

                    if len(genres) > 0 and isinstance(genres[0], dict):

                        return [g["name"] for g in genres]

                    return genres

            return [g.strip() for g in x.split(",")]

        except:

            return []

    df["genres"] = df["genres"].fillna("")

    df["genres"] = df["genres"].apply(extract_genres)

    # --------------------------------------------------------
    # Genre Encoding
    # --------------------------------------------------------

    genre_encoded = pd.DataFrame(

        mlb.transform(df["genres"]),

        columns=mlb.classes_,

        index=df.index

    )

    df = pd.concat([df, genre_encoded], axis=1)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.reindex(columns=feature_columns, fill_value=0)

    y = df["success"]

    # --------------------------------------------------------
    # Train Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y

    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------

    st.header("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.2%}")

    c2.metric("Precision", f"{precision:.2%}")

    c3.metric("Recall", f"{recall:.2%}")

    c4.metric("F1 Score", f"{f1:.2%}")

    st.divider()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.header("📌 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(

        confusion_matrix(y_test, y_pred),

        display_labels=["Failure", "Success"]

    )

    disp.plot(ax=ax)

    st.pyplot(fig)

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    st.header("📈 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    ax2.plot([0, 1], [0, 1], "--")

    ax2.set_xlabel("False Positive Rate")

    ax2.set_ylabel("True Positive Rate")

    ax2.set_title("ROC Curve")

    ax2.legend()

    st.pyplot(fig2)

    st.divider()

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    st.header("⭐ Feature Importance")

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    st.bar_chart(

        importance.set_index("Feature")

    )

    st.dataframe(

        importance,

        use_container_width=True

    )

    st.divider()

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    st.header("📄 Classification Report")

    report = classification_report(

        y_test,

        y_pred,

        output_dict=True

    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(

        report_df,

        use_container_width=True

    )

    st.divider()

    # --------------------------------------------------------
    # Model Details
    # --------------------------------------------------------

    st.header("ℹ️ Model Information")

    st.info("""

**Algorithm**

Random Forest Classifier

**Input Features**

- Budget
- Popularity
- Runtime
- Vote Average
- Genres

**Target Variable**

Movie Success

**Train-Test Split**

80% Training • 20% Testing

""")

    st.divider()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.header("📝 Interpretation")

    st.success(f"""

The Random Forest model achieved an **Accuracy of {accuracy:.2%}**.

The confusion matrix summarizes the model's predictions.

The ROC Curve evaluates the classifier's performance across different thresholds.

The Feature Importance chart highlights which variables most influence movie success prediction.

""")