import pandas as pd

# Load Dataset
df = pd.read_csv("data/movies.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

# Display basic information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

#Check Missing Values
print("\nMissing Values")
print("-" * 40)

print(df.isnull().sum())

#Check Duplicate Records
print("\nDuplicate Rows")
print("-" * 40)

print(df.duplicated().sum())


#Check Budget and Revenue
print("\nMovies with Budget = 0")
print((df["budget"] == 0).sum())

print("\nMovies with Revenue = 0")
print((df["revenue"] == 0).sum())

#Remove Invalid Movies
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

print("\nShape After Removing Invalid Records:")
print(df.shape)


#Create Success Column
df["success"] = (df["revenue"] > df["budget"]).astype(int)

print("\nSuccess Distribution")

print(df["success"].value_counts())

#Success Percentage
success_rate = df["success"].mean() * 100

print(f"\nMovie Success Rate: {success_rate:.2f}%")

#Clean Genre Column
df["genres"] = df["genres"].str.replace("|", ", ", regex=False)

#Remove Leading/Trailing Spaces
df["title"] = df["title"].str.strip()

df["genres"] = df["genres"].str.strip()

#Check Numeric Summary
print("\nSummary Statistics")

print(df.describe())

#Save the Clean Dataset
df.to_csv("data/movies_cleaned.csv", index=False)