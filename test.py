import pandas as pd
from sklearn.model_selection import train_test_split
if __name__ == "__main__":

    df = pd.read_csv("datasets/imdb_movies_preprocessed.csv", sep=";")

    # Create an empty DataFrame to store the split data
    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)

    # Get unique ratings from the DataFrame
    unique_ratings = df['rating'].unique()

    # Split each rating into 80% training and 20% testing
    for rating in unique_ratings:
        subset = df[df['rating'] == rating]
        train_subset, test_subset = train_test_split(subset, test_size=0.2, random_state=42)

        train_data = pd.concat([train_data, train_subset], ignore_index=True)
        test_data = pd.concat([test_data, test_subset], ignore_index=True)
        # train_data = train_data.append(train_subset)
        # test_data = test_data.append(test_subset)

    print(train_data.value_counts("rating"))
    print(test_data.value_counts("rating"))
    
    # Save the split data to CSV
    train_data.to_csv("datasets/imdb_movies_preprocessed_train.csv", sep=";", index=False)
    test_data.to_csv("datasets/imdb_movies_preprocessed_test.csv", sep=";", index=False)
    