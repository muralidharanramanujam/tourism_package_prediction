import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop unique identifier so it is not treated as a predictive feature
df.drop(columns=["CustomerID"], inplace=True, errors="ignore")

# NOTE: Categorical string columns (e.g., TypeofContact, Occupation, Gender,
# MaritalStatus, Designation, ProductPitched) are intentionally kept as raw
# strings so training preprocessing and inference payloads stay consistent.

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# stratify=y preserves the class balance of ProdTaken across train and test splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Features:", list(X.columns))
print("Target distribution:\n", y.value_counts(normalize=True))
