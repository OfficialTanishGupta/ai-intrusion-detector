import joblib
import torch
import torch.nn as nn
import pandas as pd
from preprocess import load_and_preprocess

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load data and scaler
X, y, scaler = load_and_preprocess("../data/network_data.csv")

# Load ML Model
ml_model = joblib.load("../models/ml_model.pkl")

# Load NN Model
nn_model = Net(X.shape[1])
nn_model.load_state_dict(torch.load("../models/nn_model.pth"))
nn_model.eval()

print("Both models loaded successfully!")

def predict(sample):
    columns = [
        'duration',
        'protocol_type',
        'src_bytes',
        'dst_bytes',
        'flag'
    ]

    sample_df = pd.DataFrame([sample], columns=columns)
    sample_scaled = scaler.transform(sample_df)

    rf_prediction = ml_model.predict(sample_scaled)[0]

    with torch.no_grad():
        sample_tensor = torch.tensor(
            sample_scaled,
            dtype=torch.float32
        )
        nn_score = nn_model(sample_tensor).item()

    print(f"NN Score: {nn_score:.4f}, RF Prediction: {'Intrusion' if rf_prediction == 1 else 'Normal'}")

sample = [50, 1, 9000, 100, 0]
predict(sample)
