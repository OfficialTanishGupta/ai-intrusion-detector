from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import load_and_preprocess

X, y, scaler = load_and_preprocess("../data/network_data.csv")

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "../models/ml_model.pkl")

print("✅ ML model trained")