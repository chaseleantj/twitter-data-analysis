import pandas as pd
import numpy as np
import data_utils
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def get_feature_vector(df, embedding_vector=True, embeddings=None, additional_features=None):
    feature_vector = None
    if embedding_vector:
        if embeddings is None:
            feature_vector = data_utils.df_extract_2darray(df, "embedding")
        else:
            feature_vector = embeddings
    if additional_features is not None:
        for feature in additional_features:
            feature_vector = np.concatenate((feature_vector, np.expand_dims(df[feature].to_numpy(), axis=1)), axis=1)
    return feature_vector

def split_data(df, train_size=0.8, val_size=0.1):
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=0)
    relative_val_size = val_size / (1.0 - train_size)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_size, random_state=0)
    return train_df, val_df, test_df

def get_features_and_labels(df, embedding_vector=True, additional_features=None, one_hot=False):
    features = get_feature_vector(df, embedding_vector, additional_features)
    if one_hot:
        labels = one_hot_encode(df['tweet type'].to_numpy(), ["Reply", "Tweet", "Thread content", "Quote", "Community"])
        # labels = pd.get_dummies(df['tweet type'], prefix='type').to_numpy()
    else:
        labels = df['tweet type'].to_numpy()
    return features, labels

def one_hot_encode(data, categories):
    one_hot_matrix = [[0]*len(categories) for _ in data]
    for i, value in enumerate(data):
        if value in categories:
            one_hot_matrix[i][categories.index(value)] = 1
        else:
            raise ValueError(f"Value {value} not in categories.")
    return np.array(one_hot_matrix)

def one_hot_to_labels(one_hot_matrix):
    labels = ["Reply", "Tweet", "Thread content", "Quote", "Community"]
    indices = np.argmax(one_hot_matrix, axis=1)
    return np.array([labels[i] for i in indices])

class Tweet_Classifier:
    def __init__(self, model, normalizer=None):
        self.model = model
        self.normalizer = normalizer
    
    def normalize(self, features):
        if self.normalizer:
            return self.normalizer(features)
        else:
            return features
    
    def fit(self, features, labels):
        self.normalizer.adapt(features)
        self.model.fit(self.normalize(features), labels)

    def base_predict(self, features, one_hot=False):
        prediction = self.model.predict(self.normalize(features))
        if one_hot:
            return one_hot_to_labels(prediction)
        else:
            return np.array(prediction)
    
    # def predict(self, df, features, one_hot=False):
    #     predictions = self.base_predict(features, one_hot)

    #     text_content = df["Tweet text"].to_numpy(dtype=str)

    #     reply_indices = np.where(np.char.startswith(text_content, "@"))[0]
    #     thread_content_indices = np.where(np.char.find(text_content, "@chaseleantj") >= 0)[0]

    #     predictions[reply_indices] = "Reply"
    #     predictions[thread_content_indices] = "Thread content"

    #     return predictions

    def predict(self, target_df, main_df, features, one_hot=False):
        target_df = target_df.reset_index()
        predictions = self.base_predict(features, one_hot)

        text_content = target_df["Tweet text"].to_numpy(dtype=str)

        reply_indices = np.where(np.char.startswith(text_content, "@"))[0]
        end_thread_content_indices = np.where(np.char.find(text_content, "@chaseleantj") >= 0)[0]

        predictions[reply_indices] = "Reply"
        predictions[end_thread_content_indices] = "Thread content"

        # If the prediction is "Thread content", make sure it is published alongside other tweets
        thread_content_indices = np.where(predictions == "Thread content")[0]

        # Convert the 'time' column to datetime if it's not already
        target_df['time'] = pd.to_datetime(target_df['time'])
        main_df['time'] = pd.to_datetime(main_df['time'])

        # Check for each "Thread content" prediction
        for idx in thread_content_indices:
            window_start = target_df.iloc[idx]['time'] - pd.Timedelta(minutes=5)
            window_end = target_df.iloc[idx]['time']

            # Subset df to rows within the 30-minute window before the current "Thread content"
            window_df = main_df[(main_df['time'] >= window_start) & (main_df['time'] <= window_end)]

            # Check if the impressions of this entry is the max of the window
            if target_df.iloc[idx]['log impressions'] == window_df['log impressions'].max():
                predictions[idx] = "Tweet"
        
        # If the prediction is "Tweet" and published at the exact same time as other tweets, make sure it has the max impressions
        # Otherwise change it to "Thread content"
        tweet_indices = np.where(predictions == "Tweet")[0]

        for idx in tweet_indices:
            window_df = main_df[main_df['time'] == target_df.iloc[idx]['time']]

            # Check if the impressions of this entry is the max of the window
            if target_df.iloc[idx]['log impressions'] != window_df['log impressions'].max():
                predictions[idx] = "Thread content"

        return predictions

    
    def get_accuracy(self, df_target, main_df, features, labels, one_hot=False):
        if one_hot:
            labels = one_hot_to_labels(labels)
        return np.sum(self.predict(df_target, main_df, features, one_hot) == labels) / len(labels)
    
    def print_wrong_predictions(self, df_target, main_df, features, labels, one_hot=False):
        text_content = df_target["Tweet text"].to_numpy(dtype=str)
        predictions = self.predict(df_target, main_df, features, one_hot)
        print("The following are the wrong predictions:")
        print(f"{'Index':<{5}}", f"{'Tweet text':>{62}}", f"{'Prediction':>{20}}", f"{'Label':>{20}}")
        print("-" * 110)

        if one_hot:
            labels = one_hot_to_labels(labels)

        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                text = text_content[i].replace("\n", " ")
                text = text[:60] + '...' if len(text) > 60 else text
                print(f"{i:<{5}} {text:<{63}}", end="")
                print(f"{predictions[i]:>{20}}", f"{labels[i]:>{20}}")

        print("-" * 110)
        num_correct = np.sum(predictions == labels)
        num_labels = len(labels)
        print("Correct predictions: " + str(num_correct) + "/" + str(num_labels))
        print("Accuracy: " + str(num_correct / num_labels * 100) + "%")


def build_model():

    months = ["may", "june", "july", "august", "september"]
    file_arr = [f"./data/processed/{month}_2023.xlsx" for month in months]

    df = data_utils.join_data(file_arr)
    embedding_paths = [f"./data/embeddings/{month}_2023_embeddings.pickle" for month in months]
    embeddings = data_utils.load_embeddings(embedding_paths)

    df = df.reset_index()
    df = data_utils.df_add_2darray(df, embeddings, "embedding")
    additional_features = ["log impressions", "existing followers"]
    is_one_hot = False

    train_df, val_df, test_df = split_data(df, train_size=0.7, val_size=0.15)
    train_features, train_labels = get_features_and_labels(train_df, additional_features=additional_features, one_hot=is_one_hot)
    val_features, val_labels = get_features_and_labels(val_df, additional_features=additional_features, one_hot=is_one_hot)
    test_features, test_labels = get_features_and_labels(test_df, additional_features=additional_features, one_hot=is_one_hot)

    model = LinearSVC(random_state=0, max_iter=1000)
    # model = LogisticRegression(random_state=0, max_iter=1000)

    # model = keras.models.load_model('models/tweet_classifier2')

    classifier = Tweet_Classifier(
        model, 
        keras.layers.Normalization()
    )

    print(train_features[:10], train_labels[:10])
    # classifier.fit(train_features, train_labels)
    # classifier.print_wrong_predictions(val_df, df, val_features, val_labels, one_hot=is_one_hot)

build_model()








# normalizer = keras.layers.Normalization()
# normalizer.adapt(train_features)

# class myCallback(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('val_loss') < 0.16):
#             print("\nValidation loss is low so cancelling training.")
#             self.model.stop_training = True

# callbacks = myCallback()

# # Build the model
# model = keras.Sequential([
#     normalizer,
#     keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
#     keras.layers.Dropout(0.1),
#     keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
#     keras.layers.Dropout(0.1),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(5, activation='softmax')
# ])

# print(model.summary())

# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.00005),
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
# )

# # Train the model
# history = model.fit(
#     train_features, train_labels,
#     epochs=100,
#     verbose=1,
#     validation_data=(val_features, val_labels),
#     callbacks=[callbacks]
# )

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch

# def plot_loss(history):
#   plt.plot(history.history['loss'], label='loss')
#   plt.plot(history.history['val_loss'], label='val_loss')
#   plt.xlabel('Epoch')
#   plt.ylabel('Loss')
#   plt.legend()
#   plt.grid(True)
#   plt.show()

# plot_loss(history)

# model.save("models/tweet_classifier2")
# print("Model saved.")


