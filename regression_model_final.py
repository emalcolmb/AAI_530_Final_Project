import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import timedelta

print("Loading dataset...")
df = pd.read_csv('amphiro_trialA_cleaned.csv')

print("Converting datetime column to datetime format...")
df['Alicante.local.date.time'] = pd.to_datetime(df['Alicante.local.date.time'])

print("Encoding the user.key column...")
df['user_key_encoded'] = df['user.key'].astype('category').cat.codes

print("Engineering features...")
# Calculate average volume per shower for each user
df['avg_volume_per_user'] = df.groupby('user.key')['volume'].transform('mean')

# Calculate frequency of showers per user
df['shower_frequency'] = df.groupby('user.key')['Alicante.local.date.time'].transform(lambda x: x.diff().dt.total_seconds().mean())

# Extract additional temporal features
df['hour_of_day'] = df['Alicante.local.date.time'].dt.hour
df['day_of_week'] = df['Alicante.local.date.time'].dt.dayofweek  # Monday=0, Sunday=6

# Prepare features and target variable
X = df[['user_key_encoded', 'hour_of_day', 'day_of_week', 'avg_volume_per_user', 'shower_frequency']]
y = df['volume']

print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Random Forest Regressor model...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("Model trained. Generating tailored predictions for each user...")

predictions = []
unique_users = df['user.key'].unique()

for user in unique_users:
    user_code = df[df['user.key'] == user]['user_key_encoded'].iloc[0]
    max_date = df[df['user.key'] == user]['Alicante.local.date.time'].max()
    avg_volume_per_user = df[df['user.key'] == user]['avg_volume_per_user'].iloc[0]
    shower_frequency = df[df['user.key'] == user]['shower_frequency'].iloc[0]

    for day in range(1, 6):  # Generate predictions for 5 consecutive days after the max date
        prediction_date = max_date + timedelta(days=day)
        day_of_week = prediction_date.dayofweek
        # Assume a random hour for simplicity; this can be refined based on historical data or other logic
        random_hour = np.random.randint(0, 24)
        
        input_features = pd.DataFrame([[user_code, random_hour, day_of_week, avg_volume_per_user, shower_frequency]],
                                      columns=['user_key_encoded', 'hour_of_day', 'day_of_week', 'avg_volume_per_user', 'shower_frequency'])
        prediction_volume = model.predict(input_features)[0]
        
        predictions.append({
            'user.key': user,
            'date': prediction_date,
            'hour': random_hour,
            'predicted_volume': prediction_volume
        })

predictions_df = pd.DataFrame(predictions)
print(f"Generated {len(predictions)} predictions tailored to users' usage history.")

# Mark original data as 'actual'
df['Type'] = 'Actual'

# Prepare the predictions DataFrame with the necessary columns to match the original DataFrame
# Assuming predictions_df already exists and contains the necessary prediction data
predictions_df['Alicante.local.date.time'] = predictions_df['date']
predictions_df['volume'] = predictions_df['predicted_volume']
predictions_df['hour_of_day'] = predictions_df['hour']
predictions_df['user.key'] = predictions_df['user.key']
# Mark prediction data as 'Predicted'
predictions_df['Type'] = 'Predicted'

# Selecting relevant columns to match the original dataframe structure (adjust columns as needed)
relevant_columns = ['user.key', 'Alicante.local.date.time', 'volume', 'hour_of_day', 'Type']
predictions_df = predictions_df[relevant_columns]

# Concatenating the original dataframe with the predictions dataframe
combined_df = pd.concat([df[relevant_columns], predictions_df], ignore_index=True)

# Now, combined_df contains both the original data and the predictions, with a 'Type' column differentiating them
print("Original data and predictions have been combined.")

# Optionally, save the combined dataframe to a new CSV file
combined_df.to_csv('combined_data_with_predictions.csv', index=False)
print("Combined data with predictions saved to 'combined_data_with_predictions.csv'.")
