from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Input features
y = iris.target  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a non-linear regression model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100)
# Fit the model to the training data
model.fit(X_train, y_train)
# Predict the target variable
y_pred = model.predict(X_test)
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
