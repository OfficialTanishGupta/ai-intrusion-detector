import joblib
import torch
from preprocess import load_and_preprocess
from train_nn import Net

# Load data (for scaler consistency)
X, y, scaler = load_and_preprocess("../data/network_data.csv")

# Load ML model
ml_model = joblib.load("../models/ml_model.pkl")

# Load NN model
nn_model = Net(X.shape[1])
nn_model.load_state_dict(torch.load("../models/nn_model.pth"))
nn_model.eval()

def predict(sample):
    sample = scaler.transform([sample])

    # ML prediction
    ml_pred = ml_model.predict_proba(sample)[0][1]

    # NN prediction
    sample_tensor = torch.tensor(sample, dtype=torch.float32)
    nn_pred = nn_model(sample_tensor).item()

    # Final hybrid score
    final = 0.5 * ml_pred + 0.5 * nn_pred

    return final

# Example test
sample = [10, 1, 1000, 500, 2]
print("Final Score:", predict(sample))