import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

url = "https://raw.githubusercontent.com/krishnaik06/Car-Price-Prediction/master/car%20data.csv"
df = pd.read_csv(url)

df = df.dropna()

# Save feature columns BEFORE encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/car_model.pkl", "wb"))

# Save columns (VERY IMPORTANT)
pickle.dump(feature_columns, open("model/features.pkl", "wb"))

print("Model + Features saved")