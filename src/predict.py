import joblib
import torch
from preprocess import load_and_preprocess
from model_nn import Net
import pandas as pd

# Load data (for scaler consistency)
X, y, scaler = load_and_preprocess("../data/network_data.csv")

# Load ML model
ml_model = joblib.load("../models/ml_model.pkl")

# Load NN model
nn_model = Net(X.shape[1])
nn_model.load_state_dict(torch.load("../models/nn_model.pth"))
nn_model.eval()

def predict(sample):
    columns = ['duration','protocol_type','src_bytes','dst_bytes','flag']
    sample_df = pd.DataFrame([sample], columns=columns)
    
    sample_scaled = scaler.transform(sample_df)

    ml_pred = ml_model.predict_proba(sample_scaled)[0][1]

    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
    nn_pred = nn_model(sample_tensor).item()

    final = 0.5 * ml_pred + 0.5 * nn_pred

    return final


# Example test
sample = [10, 1, 1000, 500, 2]
print("Final Score:", predict(sample))