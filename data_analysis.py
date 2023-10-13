import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import mplcursors
import data_utils
import statsmodels.api as sm
from sklearn.manifold import TSNE
from tensorflow import keras
from sklearn.model_selection import train_test_split

def load_data(file_arr, follower_threshold=None, duration_threshold=None, exclude_multitags=True, exclude_selected_accs=True):

    df = data_utils.join_data(file_arr)

    # Extra data processing
    if exclude_multitags:
        df = df[df["Tweet text"].str.count("@") <= 1]
    if follower_threshold:
        df = df[df["existing followers"] > follower_threshold]
    if duration_threshold:
        df = df[df["duration"] < duration_threshold]

    # # Remove the following substrings
    if exclude_selected_accs:
        substrings = ["@chaseleantj", "@icreatelife", "@4rtofficial", "@Cakedroid", "@mariswaran", "@g0rillaAI", "@foxtrotfrog", "@Arcaneship"]
        pattern = "|".join(substrings)
        df = df[df["Tweet text"].str.contains(pattern) == False]

    return df

def plot_histogram(df, target, keys=["Tweet", "Reply", "Thread content"]):
    for key in keys:
        data = df[df["tweet type"] == key][target]
        width = 0.1
        plt.hist(data, bins=np.arange(min(data), max(data) + width, width), alpha=0.5, edgecolor="black")
    plt.show(block=True)

def plot_scatter(df, x, y):
    sns.scatterplot(data=df, x=x, y=y, alpha=0.5, s=30, hue="tweet type")
    scatter =  plt.scatter(x=df[x], y=df[y], alpha=0)
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(df["Tweet text"].to_list()[sel.index])
        sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=1)
    plt.show(block=True)

def plot_regression(df, x, y):
    sns.lmplot(data=df, x=x, y=y, scatter_kws={'alpha':0.25, 's':10}, ci=None, hue="tweet type", line_kws={'alpha':0})
    scatter =  plt.scatter(data=df, x=x, y=y, alpha=0)
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(df["Tweet text"].to_list()[sel.index])
        sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=1)
    plt.show(block=True)

def ols(X, y):

    X_reg = sm.add_constant(X)
    model = sm.OLS(y, X_reg)
    results = model.fit()
    print(results.summary())

    # Print R^2
    print("R^2: ", results.rsquared)

    # Find the variance of residuals
    print("Variance of residuals: ", np.var(results.resid))

    # Plot a scatterplot of the predicted values against the actual values
    sns.scatterplot(x=results.predict(), y=y, alpha=0.5, s=10)
    plt.xlabel("predicted values")
    plt.ylabel("actual values")
    plt.title("Predicted values vs actual values")
    plt.show(block=True)

    # Plot the residuals, and a histogram of the residuals together
    fig, ax = plt.subplots(1, 2)
    sns.histplot(data=results.resid, ax=ax[0])
    ax[0].set_title("Histogram of residuals")
    ax[0].set_xlabel("residuals")
    ax[0].set_ylabel("count")

    # If X is a single column, plot the residuals against X
    if X.shape[1] == 1:
        sns.scatterplot(x=X, y=results.resid, alpha=0.5, s=10, ax=ax[1])
        ax[1].set_xlabel("x values")
        ax[1].set_ylabel("residuals")
        ax[1].set_title(f"Residuals vs x values")

    plt.show(block=True)

    # Plot a normal Q-Q plot of the residuals
    import scipy.stats as stats
    stats.probplot(results.resid, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot of residuals")
    plt.show(block=True)

    # Produce an autocorrelation plot and partial autocorrelation plot of the residuals
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, ax = plt.subplots(1, 2)
    plot_acf(results.resid, ax=ax[0])
    ax[0].set_title("ACF plot of residuals")
    ax[0].set_ylim(-0.5, 1.2)
    plot_pacf(results.resid, ax=ax[1])
    ax[1].set_title("PACF plot of residuals")
    ax[1].set_ylim(-0.5, 1.2)
    plt.show(block=True)


def plot_tsne(df, embeddings, target):

    model = TSNE(n_components=2, perplexity=5, init='random', n_iter=1000, learning_rate=100)
    tsne = model.fit_transform(embeddings).T

    plt.gca().set_facecolor("black")
    scatter = plt.scatter(x=tsne[0], y=tsne[1], s=40, alpha=0.9, cmap="magma", c=df[target], edgecolors="white", linewidths=0.2)
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(df["Tweet text"].to_list()[sel.index])
        sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=1)
    plt.show()

def predict_log_impressions_with_embeddings(embeddings, model):
    model = keras.models.load_model(f'models/{model}')
    predictions = model.predict(embeddings)
    return predictions

months = ["may", "june", "july", "august", "september"]
file_arr = [f"data/processed/{month}_2023.xlsx" for month in months]

df = load_data(file_arr, follower_threshold=None, duration_threshold=None, exclude_multitags=False, exclude_selected_accs=False)
embedding_paths = [f"data/embeddings/{month}_2023_embeddings.pickle" for month in months]
embeddings = data_utils.load_embeddings(embedding_paths)

# df, embeddings = data_utils.filter_df_and_embeddings(df, subset="Tweet", embeddings=embeddings)
log_existing_followers = np.log(df["existing followers"].to_numpy())
embeddings = np.concatenate((embeddings, log_existing_followers.reshape(-1, 1)), axis=1)
ols(embeddings, df["log impressions"])


# plot_histogram(df, target="unexplained")
# plot_scatter(df, "log impressions", "user profile clicks ratio")
# plot_regression(df, "log existing followers", "unexplained")
# plot_tsne(df, embeddings, "user profile clicks ratio")

