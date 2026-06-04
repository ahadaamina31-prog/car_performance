import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Load dataset
url = "https://raw.githubusercontent.com/krishnaik06/Car-Price-Prediction/master/car%20data.csv"
df = pd.read_csv(url)

# Remove missing values
df = df.dropna()

# ❌ REMOVE STRING COLUMN (VERY IMPORTANT FIX)
df = df.drop("Car_Name", axis=1)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save folder
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/car_model.pkl", "wb"))

# Save feature columns (VERY IMPORTANT)
pickle.dump(X.columns, open("model/features.pkl", "wb"))

print("Model + Features saved successfully")