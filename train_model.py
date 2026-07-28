import pandas as pd
import ast
import joblib
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# -----------------------------
# Convert genres
# -----------------------------
def extract_genres(x):
    try:
        genres = ast.literal_eval(x)
        return [g["name"] for g in genres]
    except:
        return []

df["genres"] = df["genres"].fillna("[]")
df["genres"] = df["genres"].apply(extract_genres)

# -----------------------------
# Encode Genres
# -----------------------------
mlb = MultiLabelBinarizer()

genre_df = pd.DataFrame(
    mlb.fit_transform(df["genres"]),
    columns=mlb.classes_,
    index=df.index
)

df = pd.concat([df, genre_df], axis=1)

# -----------------------------
# Features
# -----------------------------
feature_columns = [
    "budget",
    "popularity",
    "runtime",
    "vote_average"
]

feature_columns.extend(list(mlb.classes_))

X = df[feature_columns]

y = df["success"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("=" * 50)
print("Model Accuracy :", round(accuracy * 100, 2), "%")
print("=" * 50)

# -----------------------------
# Save Files
# -----------------------------
joblib.dump(model, MODEL_DIR / "random_forest.pkl")
joblib.dump(mlb, MODEL_DIR / "genre_encoder.pkl")
joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

print("Model Saved Successfully")