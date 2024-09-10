import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load Datasets
df_1_sep = pd.read_csv("D:\SIH2024\Datasets\dataset1.csv")
df_2_sep = pd.read_csv("D:\SIH2024\Datasets\dataset2.csv")
df_3_sep = pd.read_csv("D:\SIH2024\Datasets\dataset3.csv") 

# Step 2: Add Date Columns
df_1_sep['Date'] = '01-09-2024'
df_2_sep['Date'] = '02-09-2024'
df_3_sep['Date'] = '03-09-2024'

# Step 3: Combine Datasets
combined_df = pd.concat([df_1_sep, df_2_sep, df_3_sep], ignore_index=True)

# Convert Date column to datetime format
combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%d-%m-%Y')

# Step 4: Handle Missing Values
combined_df['Arrivals'] = pd.to_numeric(combined_df['Arrivals'], errors='coerce')
combined_df['Minimum Prices'] = pd.to_numeric(combined_df['Minimum Prices'], errors='coerce')
combined_df['Maximum Prices'] = pd.to_numeric(combined_df['Maximum Prices'], errors='coerce')
combined_df['Modal Prices'] = pd.to_numeric(combined_df['Modal Prices'], errors='coerce')

# Fill missing values only for numeric columns
combined_df.fillna(combined_df.select_dtypes(include='number').mean(), inplace=True)

# Step 5: Take user input for the crop variety
crop_variety = input("Enter the crop variety you want predictions for: ")

# Filter data for the specific crop
crop_data = combined_df[combined_df['Variety'].str.lower() == crop_variety.lower()]

# Step 6: Integrate with Prediction Code
X = crop_data[['Arrivals', 'Minimum Prices', 'Maximum Prices']]
y = crop_data['Modal Prices']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for {crop_variety}: {mse}")

# Provide predictions for the entire dataset
predicted_prices = model.predict(X_scaled)

# Add predictions to the DataFrame
crop_data['Predicted Prices'] = predicted_prices

# Step 7: Calculate Averages for Each Day
average_prices = crop_data.groupby('Date').agg({
    'Minimum Prices': 'mean',
    'Maximum Prices': 'mean',
    'Modal Prices': 'mean',
    'Predicted Prices': 'mean'
}).reset_index()

# Step 8: Display Price Table
print(average_prices)

# Step 9: Generate the Trend Chart with Averaged Prices
plt.figure(figsize=(10, 6))

# Plotting the average minimum prices
plt.plot(average_prices['Date'], average_prices['Minimum Prices'], marker='o', linestyle='-', color='green', linewidth=2, label='Avg. Minimum Prices')

# Plotting the average maximum prices
plt.plot(average_prices['Date'], average_prices['Maximum Prices'], marker='o', linestyle='-', color='red', linewidth=2, label='Avg. Maximum Prices')

# Plotting the average modal prices
plt.plot(average_prices['Date'], average_prices['Modal Prices'], marker='o', linestyle='-', color='blue', linewidth=2, label='Avg. Modal Prices')

# Plotting the average predicted prices
plt.plot(average_prices['Date'], average_prices['Predicted Prices'], marker='o', linestyle='--', color='orange', linewidth=2, label='Avg. Predicted Prices')

# Title and labels
plt.title(f"Average Price Trend for {crop_variety}", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=14)
plt.ylabel("Average Price (₹ per Quintal)", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Grid lines for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adding legend
plt.legend(fontsize=12)

# Additional styling
plt.tight_layout()
plt.show()

# Optional: Save the combined dataset with predictions to a new CSV
average_prices.to_csv(f"{crop_variety}_average_predictions.csv", index=False)
