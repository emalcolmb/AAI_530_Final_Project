import pandas as pd
import matplotlib.pyplot as plt

# Define column names
#column_names = [
#    'user.key', 'device.key', 'session.id', 'Alicante.local.date.time',
#    'volume', 'temperature', 'energy', 'flow', 'duration', 'history', 'mobile.os'
#]

# Read the CSV file into a pandas DataFrame, skipping the first row
#df = pd.read_csv('amphiro_trialA.csv', header=None, names=['data'], skiprows=1)

# Split the 'data' column by semicolon and expand into separate columns
#split_data = df['data'].str.split(';', expand=True)

# Update the column names
##split_data.columns = column_names

# Save the modified DataFrame back to the CSV file without writing the header row again
#split_data.to_csv('amphiro_trialA_modified.csv', index=False)


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('amphiro_trialA_modified.csv')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For more visually appealing plots
import time  # Import the time module for the sleep function

# Function to print messages with a pause
def print_with_pause(message):
    print(message)
    time.sleep(1.5)  # Pause for 1.5 seconds

# Setting up seaborn for better aesthetics
print_with_pause("Setting up seaborn for visually appealing plots.")
sns.set_style("darkgrid")

# Reading the CSV file into a pandas DataFrame
print_with_pause("Reading the modified CSV file into a pandas DataFrame.")
df = pd.read_csv('amphiro_trialA_modified.csv')
print_with_pause("CSV file successfully read into DataFrame.")

# Parsing the date-time column with an explicit format
print_with_pause("Parsing the 'Alicante.local.date.time' column into datetime format.")
df['Alicante.local.date.time'] = pd.to_datetime(df['Alicante.local.date.time'], format='%d/%m/%Y %H:%M:%S', dayfirst=True)
print_with_pause("Date-time column successfully parsed.")

# Forward filling missing values
print_with_pause("Forward filling missing values to handle missing data.")
df.ffill(inplace=True)
print_with_pause("Missing values handled via forward fill.")

# Specify the file name for the cleaned data
cleaned_file_name = 'amphiro_trialA_cleaned.csv'

# Write the cleaned DataFrame to a new CSV file, without the index
df.to_csv(cleaned_file_name, index=False)

print("Cleaned data has been successfully written to:", cleaned_file_name)


# Preparing for data visualization by handling outlier detection and preprocessing steps
print_with_pause("Assuming outlier detection and preprocessing steps are correctly handled above.")

# Visualization section starts here
print_with_pause("Starting data visualization with enhanced plots.")

# Creating a box plot for each numerical variable
print_with_pause("Creating a box plot for numerical variables.")
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.select_dtypes(include='number'), orient="h", palette="Set2")
plt.title('Box plot of numerical variables')
plt.tight_layout()
plt.show()
print_with_pause("Box plot created successfully.")

# Creating a scatter plot for volume vs. temperature
print_with_pause("Creating a scatter plot for volume vs. temperature.")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='volume', y='temperature', alpha=0.6, edgecolor=None, palette="coolwarm")
plt.title('Scatter plot of volume vs. temperature')
plt.xlabel('Volume')
plt.ylabel('Temperature')
plt.show()
print_with_pause("Scatter plot created successfully.")

# Creating a heatmap for correlations among numerical variables
print_with_pause("Creating a heatmap to show correlations among numerical variables.")
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()  # Calculating the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
print_with_pause("Heatmap of correlations created successfully.")

# Function to print messages with a pause
def print_with_pause(message):
    print(message)
    time.sleep(1.5)  # Pause for 1.5 seconds

# Setting up seaborn for better aesthetics
sns.set_style("darkgrid")

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('amphiro_trialA_modified.csv')

# Parsing the date-time column with an explicit format
df['Alicante.local.date.time'] = pd.to_datetime(df['Alicante.local.date.time'], format='%d/%m/%Y %H:%M:%S', dayfirst=True)

# Forward filling missing values
df.ffill(inplace=True)

# Calculate and print relevant statistics
average_volume = df['volume'].mean()
max_volume = df['volume'].max()
min_volume = df['volume'].min()
average_duration = df['duration'].mean()
correlation_temp_volume = df[['temperature', 'volume']].corr().iloc[0, 1]

print_with_pause(f"Average water volume usage per shower session: {average_volume:.2f} liters")
print_with_pause(f"Maximum water volume usage in a single shower session: {max_volume:.2f} liters")
print_with_pause(f"Minimum water volume usage in a single shower session: {min_volume:.2f} liters")
print_with_pause(f"Average shower duration: {average_duration:.2f} seconds")
print_with_pause(f"Correlation between water volume and temperature: {correlation_temp_volume:.2f}")



