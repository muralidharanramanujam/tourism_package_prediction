import os
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"
os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Drop stray index column if exported from a previous notebook/CSV save
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    df.to_csv(RAW_PATH, index=False)

# Validate that the expected feature columns and target are present
expected_columns = [
    "CustomerID",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch",
    "ProdTaken",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
