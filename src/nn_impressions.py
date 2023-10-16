from tensorflow import keras
import pandas as pd
import numpy as np
import data_utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def get_feature_vector(df, embedding_vector=True, additional_features=None):
    feature_vector = None
    if embedding_vector:
        feature_vector = data_utils.df_extract_2darray(df, "embedding")
    if additional_features is not None:
        for feature in additional_features:
            feature_vector = np.concatenate((feature_vector, np.expand_dims(df[feature].to_numpy(), axis=1)), axis=1)
    return feature_vector

def split_data(df, train_size=0.8, val_size=0.1):
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=100)
    if val_size == 0:
        return train_df, None, temp_df
    else:
        relative_val_size = val_size / (1.0 - train_size)
        val_df, test_df = train_test_split(temp_df, train_size=relative_val_size, random_state=100)
        return train_df, val_df, test_df

def get_all_features(df, embedding_vector=True, additional_features=None):
    features = get_feature_vector(df, embedding_vector, additional_features)
    # tweet_type = one_hot_encode(df['tweet type'].to_numpy(), ["Reply", "Tweet", "Thread content", "Quote", "Community"])
    # Concatenate the tweet type to the features
    # features = np.concatenate((features, tweet_type), axis=1)
    return features

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

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Mean squared error [log impressions]')
  plt.legend()
  plt.grid(True)
  plt.show()

def get_r2(labels, predictions):
    return 1 - np.sum((labels - predictions)**2) / np.sum((labels - np.mean(labels))**2)


months = ["may", "june", "july", "august", "september"]
file_arr = [f"data/processed/{month}_2023.xlsx" for month in months]

df = data_utils.join_data(file_arr)
embedding_paths = [f"data/embeddings/{month}_2023_embeddings.pickle" for month in months]
embeddings = data_utils.load_embeddings(embedding_paths)

df = data_utils.df_add_2darray(df, embeddings, "embedding")
additional_features = []
target = ["log impressions"]

train_df, val_df, test_df = split_data(df, train_size=0.7, val_size=0.15)

train_features = get_all_features(train_df, embedding_vector=True, additional_features=additional_features)
val_features = get_all_features(val_df, embedding_vector=True, additional_features=additional_features)
test_features = get_all_features(test_df, embedding_vector=True, additional_features=additional_features)

train_labels = train_df[target].to_numpy()
val_labels = val_df[target].to_numpy()
test_labels = test_df[target].to_numpy()

# Normalize the data
normalizer = keras.layers.Normalization()
normalizer.adapt(train_features)

# Build the model
model = keras.Sequential([
    normalizer,
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1)
])

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),
    loss='mean_squared_error',
)

# Train the model
history = model.fit(
    train_features, train_labels,
    epochs=150,
    batch_size=32,
    verbose=1,
    validation_data=(val_features, val_labels)
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_loss(history)

# Calculate the R^2 of the fit
val_predictions = model.predict(val_features)
test_predictions = model.predict(test_features)
print("R^2 of training set: ", get_r2(val_labels, val_predictions))
print("R^2 of testing set: ", get_r2(test_labels, test_predictions))
print("MSE of training set: ", np.mean((val_labels - val_predictions)**2))
print("MSE of testing set: ", np.mean((test_labels - test_predictions)**2))

# Plot the predictions vs. the actual values
plt.scatter(val_predictions, val_labels)
plt.plot([min(val_labels), max(val_labels)], [min(val_labels), max(val_labels)], color='red')
plt.xlabel("Predicted log impressions")
plt.ylabel("Actual log impressions")
plt.show()

# model.save("models/impressions_predictor_september_200")
# print("Model saved.")

