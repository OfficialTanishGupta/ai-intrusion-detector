import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess
from model_nn import Net

X, y, _ = load_and_preprocess("../data/network_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

model = Net(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
for epoch in range(50):
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted = torch.argmax(predictions, dim=1)
    
    accuracy = accuracy_score(
        y_test.numpy(),
        predicted.numpy()
    )

print(f"\n✅ NN Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "../models/nn_model.pth")
print("✅ NN model saved to ../models/nn_model.pth")
