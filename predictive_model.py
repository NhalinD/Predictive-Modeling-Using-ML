import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['Hours']]
y = data['Marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

# Predict for new input
hours = [[5]]
predicted_marks = model.predict(hours)

print("Predicted Marks for 5 Hours Study:", predicted_marks[0])

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.show()