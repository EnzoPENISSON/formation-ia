import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



def MoviesOrSeries(year):
    # Nettoie les caractères inutiles et vérifie si l'année contient un tiret
    year = year.replace('–', '-')
    return 1.0 if '-' in year else 0.0

def preprocess_data(data):
    # Encode le type dans GENRE_ENCODED basé sur la colonne YEAR
    for i in range(len(data)):
        data.loc[i, 'GENRE_ENCODED'] = MoviesOrSeries(data.loc[i, 'YEAR'])

    # Convertit la colonne VOTES en float après suppression des virgules
    data['VOTES'] = data['VOTES'].replace({',': ''}, regex=True).astype(float)

    return data


def predict_rating(data):
    # Preprocess data
    data = preprocess_data(data)
    print(data.head(10))
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

