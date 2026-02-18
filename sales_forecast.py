import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("sales.csv")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Create time features
data['Day_Number'] = (data['Date'] - data['Date'].min()).dt.days

# Features and Target
X = data[['Day_Number']]
y = data['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Plot results
plt.figure()
plt.plot(data['Date'], data['Sales'], label="Actual Sales")
plt.plot(data['Date'].iloc[-len(predictions):], predictions, label="Predicted Sales")
plt.legend()
plt.title("Sales Forecast")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()