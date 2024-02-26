import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta

print("Loading dataset...")
df = pd.read_csv('amphiro_trialA_cleaned.csv')  # Update this path

print("Converting datetime column to datetime format...")
df['Alicante.local.date.time'] = pd.to_datetime(df['Alicante.local.date.time'])

print("Encoding the user.key column...")
df['user_key_encoded'] = df['user.key'].astype('category').cat.codes

print("Engineering features...")
# Extract temporal features
df['hour_of_day'] = df['Alicante.local.date.time'].dt.hour
df['day_of_week'] = df['Alicante.local.date.time'].dt.dayofweek  # Monday=0, Sunday=6

# Prepare features and target variable
X = df[['user_key_encoded', 'hour_of_day', 'day_of_week']].values
y = df['temperature'].values.reshape(-1, 1)

# Scaling features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)

# Define the PyTorch model
class TemperatureModel(nn.Module):
    def __init__(self, input_size):
        super(TemperatureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = TemperatureModel(X_train.shape[1])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training the model
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generating predictions for 5 consecutive days after the max date for each user
predictions = []
unique_users = df['user.key'].unique()

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient tracking
    for user in unique_users:
        user_code = df[df['user.key'] == user]['user_key_encoded'].iloc[0]
        max_date = df[df['user.key'] == user]['Alicante.local.date.time'].max()

        for day in range(1, 6):  # Generate predictions for 5 consecutive days
            prediction_date = max_date + timedelta(days=day)
            day_of_week = prediction_date.dayofweek
            random_hour = np.random.randint(0, 24)

            input_features = torch.Tensor([[user_code, random_hour, day_of_week]])
            input_features_scaled = torch.Tensor(scaler.transform(input_features))
            predicted_temperature = model(input_features_scaled).item()

            predictions.append({
                'user.key': user,
                'date': prediction_date.strftime('%Y-%m-%d'),
                'hour': random_hour,
                'predicted_temperature': predicted_temperature
            })

predictions_df = pd.DataFrame(predictions)
print(f"Generated {len(predictions)} predictions tailored to users' preferences.")

# Optionally, save the predictions to a CSV file
predictions_df.to_csv('temperature_predictions.csv', index=False)
print("Temperature predictions saved to 'temperature_predictions.csv'.")
