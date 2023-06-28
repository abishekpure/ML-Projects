import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('C:\\Users\\Abishek Prakash\\Desktop\\song_data.csv\\song_data.csv')

# Select the features to use
features = ['danceability', 'energy', 'loudness', 'speechiness', 'audio_valence', 'tempo']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['song_popularity'], test_size=0.2, random_state=42)

# Create the decision tree regressor model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Predict the popularity of the test set
predictions = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)

# Print the mean squared error
print(mse)

# Print the accuracy of the model
print(model.score(X_test, y_test))