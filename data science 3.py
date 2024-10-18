
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
movie_data = pd.read_csv('C:\\Users\\Gopinath\\Downloads\\Tamil_movies_dataset - Tamil_movies_dataset.csv')

# Displaying the column names and checking for missing values
print("Column names:", movie_data.columns)
print(movie_data.isnull().sum())

# Filling missing values with the mean of the respective column
movie_data['PeopleVote'].fillna(movie_data['PeopleVote'].mean(), inplace=True)
movie_data['Hero_Rating'].fillna(movie_data['Hero_Rating'].mean(), inplace=True)

# Create a Success column (1 = success, 0 = failure) based on Rating > 7 and PeopleVote > 1000
movie_data['Success'] = ((movie_data['Rating'] > 7) & (movie_data['PeopleVote'] > 1000)).astype(int)

# Checking the distribution of Success
print("\nSuccess distribution:")
print(movie_data['Success'].value_counts())

# Features for prediction: Rating, PeopleVote, Hero_Rating
X = movie_data[['Rating', 'PeopleVote', 'Hero_Rating']]
y = movie_data['Success']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)

# Adding predictions to the original movie data
movie_data['Predicted_Success'] = model.predict(X[['Rating', 'PeopleVote', 'Hero_Rating']])

# Display the movie name, rating, people votes, hero rating, and success prediction
print("\nMovie success predictions:")
print(movie_data[['MovieName', 'Rating', 'PeopleVote', 'Hero_Rating', 'Predicted_Success']].head())

# Plot histograms for analysis
# Histogram for Ratings
plt.figure(figsize=(10,6))
plt.hist(movie_data['Rating'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Movies')
plt.show()

# Histogram for People Votes
plt.figure(figsize=(10,6))
plt.hist(movie_data['PeopleVote'], bins=20, color='green', edgecolor='black')
plt.title('Distribution of People Votes')
plt.xlabel('People Votes')
plt.ylabel('Number of Movies')
plt.show()

# Histogram for Hero Ratings
plt.figure(figsize=(10,6))
plt.hist(movie_data['Hero_Rating'], bins=20, color='red', edgecolor='black')
plt.title('Distribution of Hero Ratings')
plt.xlabel('Hero Rating')
plt.ylabel('Number of Movies')
plt.show()

# Histogram for Success vs Failure count
plt.figure(figsize=(10,6))
success_counts = movie_data['Predicted_Success'].value_counts()
plt.bar(success_counts.index, success_counts.values, color=['red', 'green'])
plt.title('Success vs Failure Count')
plt.xlabel('Success (1) / Failure (0)')
plt.ylabel('Number of Movies')
plt.show()
