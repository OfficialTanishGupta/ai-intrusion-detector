import random
import time
import torch
import joblib
import pandas as pd

from preprocess import load_and_preprocess
from model_nn import Net

# Load scaler
X, y, scaler = load_and_preprocess("../data/network_data.csv")

# Load ML model
ml_model = joblib.load("../models/ml_model.pkl")

# Load NN model
nn_model = Net(X.shape[1])

nn_model.load_state_dict(
    torch.load("../models/nn_model.pth")
)

nn_model.eval()

# Attack labels
labels = {
    0: "dos",
    1: "normal",
    2: "probe",
    3: "r2l",
    4: "u2r"
}

print("\n🚀 Real-Time Intrusion Detection Started...\n")

while True:

    # Simulate packet
    sample = [
        random.randint(1, 100),
        random.randint(0, 2),
        random.randint(100, 10000),
        random.randint(0, 3000),
        random.randint(0, 2)
    ]

    columns = [
        'duration',
        'protocol_type',
        'src_bytes',
        'dst_bytes',
        'flag'
    ]

    sample_df = pd.DataFrame(
        [sample],
        columns=columns
    )

    sample_scaled = scaler.transform(sample_df)

    # ML prediction
    rf_prediction = ml_model.predict(
        sample_scaled
    )[0]

    # NN prediction
    sample_tensor = torch.tensor(
        sample_scaled,
        dtype=torch.float32
    )

    nn_prediction = nn_model(sample_tensor)

    predicted_class = torch.argmax(
        nn_prediction,
        dim=1
    ).item()

    print("===================================")
    print(f"📦 Packet: {sample}")

    print(f"🌲 RF Prediction: {labels[rf_prediction]}")

    print(f"🧠 NN Prediction: {labels[predicted_class]}")

    # Alert system
    if labels[predicted_class] != "normal":

        print("🚨 ALERT: Intrusion Detected!")

    else:

        print("✅ Normal Traffic")

    time.sleep(2)