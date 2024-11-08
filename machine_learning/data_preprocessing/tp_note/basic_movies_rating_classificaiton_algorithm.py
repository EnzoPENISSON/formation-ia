import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def MoviesOrSeries(year):
    # Nettoie les caractères inutiles et vérifie si l'année contient un tiret
    year = year.replace('–', '-')
    return 1.0 if '-' in year else 0.0

def preprocess_data(data):
    # Encode the type in 'GENRE_ENCODED' based on the 'YEAR' column
    data['GENRE_ENCODED'] = data['YEAR'].apply(MoviesOrSeries)

    # Convert 'VOTES' to float after removing commas
    data['VOTES'] = data['VOTES'].replace({',': ''}, regex=True).astype(float)

    # Calculate separate mean and standard deviation for movies and series
    movie_runtime_mean = data.loc[data['GENRE_ENCODED'] == 0.0, 'RunTime'].mean()
    movie_runtime_std = data.loc[data['GENRE_ENCODED'] == 0.0, 'RunTime'].std()

    series_runtime_mean = data.loc[data['GENRE_ENCODED'] == 1.0, 'RunTime'].mean()
    series_runtime_std = data.loc[data['GENRE_ENCODED'] == 1.0, 'RunTime'].std()

    # Define a function to generate random runtime within reasonable bounds
    def generate_runtime(mean, std, min_value=60, max_value=300):
        runtime = np.random.normal(mean, std)
        return max(min_value, min(runtime, max_value))  # Clamp value within min and max

    # Fill missing 'RunTime' values for movies
    data.loc[data['RunTime'].isnull() & (data['GENRE_ENCODED'] == 0.0), 'RunTime'] = \
        data[data['RunTime'].isnull() & (data['GENRE_ENCODED'] == 0.0)].apply(
            lambda x: generate_runtime(movie_runtime_mean, movie_runtime_std), axis=1
        )

    # Fill missing 'RunTime' values for series
    data.loc[data['RunTime'].isnull() & (data['GENRE_ENCODED'] == 1.0), 'RunTime'] = \
        data[data['RunTime'].isnull() & (data['GENRE_ENCODED'] == 1.0)].apply(
            lambda x: generate_runtime(series_runtime_mean, series_runtime_std), axis=1
        )

    return data


def predict_rating(data):
    # Preprocess data
    data = preprocess_data(data)
    # Selecting features and target variable
    X = data[['RunTime', 'VOTES', 'GENRE_ENCODED']]
    y = data['RATING']

    # Handle missing values in the target variable
    X = X[~y.isna()]
    y = y[~y.isna()]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    # Calculate accuracy metrics

    mse = mean_squared_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    return predictions, y_test, r2, mse


# Load data from a CSV file
file_path = '../../../data/release/movies_cleaned.csv'  # Replace with your file path
df = pd.read_csv(file_path)


# Running the prediction
predictions, y_test, mse, r2 = predict_rating(df)
print("Predictions:", predictions)
print("Actual Ratings:", y_test.values)
print("Mean Squared Error:", mse)
print("R-squared (Accuracy):", r2)

