import pandas as pd
import numpy as np
import data_utils
import tweet_classification
from pathlib import Path
from tensorflow import keras

def derive_new_columns(df):

    # Derive new columns
    df["text length"] = df["Tweet text"].str.len()
    df["like ratio"] = df["likes"] / df["impressions"]
    df["retweet ratio"] = df["retweets"] / df["impressions"]
    df["reply ratio"] = df["replies"] / df["impressions"]
    df["user profile clicks ratio"] = df["user profile clicks"] / df["impressions"]
    df["log existing followers"] = np.log(df["existing followers"] + 1)
    df["log impressions"] = np.log(df["impressions"] + 1)
    df["log retweets"] = np.log(df["retweets"] + 1)
    df["log likes"] = np.log(df["likes"] + 1)

    # Convert the 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Find the earliest and latest time
    # earliest_time = df['time'].min()
    earliest_time = pd.to_datetime("2023-05-12 14:44:00+00:00")
    latest_time = df['time'].max()
    df['duration'] = (df['time'] - earliest_time).dt.total_seconds() / (24 * 60 * 60)

    return df


def map_followers(df, follower_df):
    # Convert the datetime column in both dataframes to date
    df['date'] = pd.to_datetime(df['time']).dt.date
    follower_df['date'] = pd.to_datetime(follower_df['date']).dt.date

    # Set the date column as index in both dataframes
    df.set_index('date', inplace=True)
    follower_df.set_index('date', inplace=True)

    # Create the new column 'existing followers' in df by mapping the 'followers' column in follower_df
    df['existing followers'] = df.index.map(follower_df['followers'])

    # Check if there are any NaN values in the 'existing followers' column
    if df['existing followers'].isna().any():
        print("Warning: Some tweet dates don't have corresponding follower data. Using the last available follower count for these dates.")
        df['existing followers'].fillna(method='ffill', inplace=True)

    # Reset the index in df
    df.reset_index(inplace=True)

    return df

def categorize_tweets(df, embeddings):

    if 'tweet type' in df.columns:
        print("Warning: There is already a column called 'tweet type'. Are you sure you want to overwrite it? (Y/N)")
        choice = str(input(">> "))

        if choice not in ['y', 'Y']:
            print("Aborting...")
            return df
        
    df.reset_index(inplace=True)
    additional_features = ["log impressions", "existing followers"]

    features = tweet_classification.get_feature_vector(df, embeddings=embeddings, additional_features=additional_features)
    model = keras.models.load_model('models/tweet_classifier2')
    classifier = tweet_classification.Tweet_Classifier(model)

    predictions = classifier.predict(target_df=df, main_df=df, features=features, one_hot=True)
    
    df['tweet type'] = predictions
    print("Successfully categorized tweets.")
    return df

def main():
    month = "september"
    df = pd.read_excel(f"./data/raw/raw_{month}.xlsx")
    follower_df = pd.read_excel("./data/followers.xlsx")

    embedding_path = f"./data/embeddings/{month}_2023_embeddings.pickle"
    if Path(embedding_path).is_file():
        print("Loading embeddings...")
        embeddings = data_utils.load_embeddings(embedding_path)
    else:
        print("Embeddings not found. Generating embeddings...")
        embeddings = data_utils.get_embeddings(df, target="Tweet text")
        data_utils.save_embeddings(embeddings, f"./data/embeddings/{month}_2023_embeddings.pickle")

    df = map_followers(df, follower_df)
    df = derive_new_columns(df)
    df = categorize_tweets(df, embeddings)

    df.to_csv(f"./data/processed/{month}_2023_new.xlsx", index=False)
    print(f"Successfuly saved cleaned data.")

main()