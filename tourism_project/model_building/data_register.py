import os
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"
os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Drop stray index column if present so downstream steps see clean columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    df.to_csv(RAW_PATH, index=False)

# Validate that all expected columns and target are present before registering
expected_columns = [
    "CustomerID",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
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
