import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# =========================
# 1. LOAD DATASET ONLINE
# =========================
url = "https://raw.githubusercontent.com/krishnaik06/Car-Price-Prediction/master/car%20data.csv"

df = pd.read_csv(url)

# =========================
# 2. CLEAN DATA
# =========================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# =========================
# 3. CONVERT CATEGORICAL DATA
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# 4. SPLIT DATA
# =========================
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
print("Model Accuracy:", model.score(X_test, y_test))

# =========================
# 6. SAVE MODEL
# =========================
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/car_model.pkl", "wb"))

print("Model saved successfully!")