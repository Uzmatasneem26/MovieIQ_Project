#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# Load cleaned dataset
df = pd.read_csv("data/movies_cleaned.csv")

#Dataset Overview
print("="*60)
print("Dataset Overview")
print("="*60)

print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSummary Statistics")
print(df.describe())

#Success Distribution
plt.figure(figsize=(6,5))

sns.countplot(x="success", data=df)

plt.title("Movie Success Distribution")
plt.xlabel("Success")
plt.ylabel("Count")

plt.savefig("assets/success_distribution.png")

plt.show()

#Budget vs Revenue Scatter Plot
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="budget",
    y="revenue",
    hue="success"
)

plt.title("Budget vs Revenue")

plt.savefig("assets/budget_vs_revenue.png")

plt.show()

#Interpretation

#Movies with larger budgets generally generate higher revenues,
# although there is considerable variation. Successful 
# movies are concentrated where revenue exceeds budget.

#Genre Distribution
genre_counts = (
    df["genres"]
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)

plt.figure(figsize=(12,6))

genre_counts.plot(kind="bar")

plt.title("Genre Distribution")

plt.ylabel("Number of Movies")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("assets/genre_distribution.png")

plt.show()

#Genre Success Rate
genre_success = (
    df.assign(genres=df["genres"].str.split(","))
      .explode("genres")
)

genre_success["genres"] = genre_success["genres"].str.strip()

genre_success = (
    genre_success
    .groupby("genres")["success"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,6))

genre_success.plot(kind="bar", color="green")

plt.ylabel("Success Rate")

plt.title("Success Rate by Genre")

plt.tight_layout()

plt.savefig("assets/genre_success.png")

plt.show()

#Popularity Distribution
plt.figure(figsize=(10,6))

sns.histplot(df["popularity"], bins=30)

plt.title("Popularity Distribution")

plt.savefig("assets/popularity_distribution.png")

plt.show()

#Runtime Distribution
plt.figure(figsize=(10,6))

sns.histplot(df["runtime"], bins=25)

plt.title("Runtime Distribution")

plt.savefig("assets/runtime_distribution.png")

plt.show()

#Vote Average Distribution
plt.figure(figsize=(10,6))

sns.histplot(df["vote_average"], bins=20)

plt.title("Vote Average Distribution")

plt.savefig("assets/vote_average_distribution.png")

plt.show()

#Correlation Heatmap
plt.figure(figsize=(8,6))

corr = df.select_dtypes(include="number").corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig("assets/heatmap.png")

plt.show()

#Success vs Popularity
plt.figure(figsize=(8,6))

sns.boxplot(
    x="success",
    y="popularity",
    data=df
)

plt.title("Popularity vs Movie Success")

plt.savefig("assets/popularity_success.png")

plt.show()

#Success vs Vote Average
plt.figure(figsize=(8,6))

sns.boxplot(
    x="success",
    y="vote_average",
    data=df
)

plt.title("Vote Average vs Movie Success")

plt.savefig("assets/vote_success.png")

plt.show()

#