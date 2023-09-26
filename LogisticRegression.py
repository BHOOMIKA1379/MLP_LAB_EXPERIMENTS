import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
df = pd.read_csv('bottle.csv', low_memory=False)
df_binary = df[['Salnty', 'T_degC']]
df_binary.columns = ['Sal', 'Temp']

# Display the first 5 rows
print(df_binary.head())

# Plot the Scatter plot to check the relationship between Sal and Temp
sns.lmplot(x="Sal", y="Temp", data=df_binary, order=2, ci=None)
plt.show()

# Eliminate NaN or missing input numbers
df_binary_copy = df_binary.copy()
df_binary_copy.fillna(method='pad', inplace=True)

# Define the temperature threshold for 'hot' and 'cold' classes
threshold = 15

# Create a new column 'TempClass' to represent the classes 'cold' and 'hot'
df_binary_copy['TempClass'] = np.where(df_binary_copy['Temp'] < threshold, 'cold', 'hot')

# Map 'cold' and 'hot' classes to numerical labels (0 and 1)
class_mapping = {'cold': 0, 'hot': 1}
df_binary_copy['TempClass'] = df_binary_copy['TempClass'].map(class_mapping)

# Extract the feature and target data
X = df_binary_copy['Sal'].values.reshape(-1, 1)
y = df_binary_copy['TempClass'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the LogisticRegression model
regr = LogisticRegression()

# Fit the model on the training data
regr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regr.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)
