import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import mplcursors
import data_utils
from sklearn.manifold import TSNE

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

def plot_histogram(df, keys=["Tweet", "Reply", "Thread content"]):
    for key in keys:
        data = df[df["tweet type"] == key]["log impressions"]
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

def ols(df, target):
    # Select the following tweets
    df_target = df[df["tweet type"] == target]

    # Perform linear regression on df_replies with ols
    import statsmodels.api as sm
    X = df_target["log existing followers"].to_numpy()
    y = df_target["log impressions"].to_numpy()

    # Write X and y as columns of a dataframe to a text file
    df_target[["log existing followers", "log impressions"]].to_csv("data/ols_data.csv", index=False)

    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    # Find the variance of residuals
    print("Variance of residuals: ", np.var(results.resid))

    # Print equation of line
    print("y = {} + {} * x".format(results.params[0], results.params[1]))

    # Plot the residuals, and a histogram of the residuals together
    fig, ax = plt.subplots(1, 2)
    sns.histplot(data=results.resid, ax=ax[0])
    ax[0].set_title("Histogram of residuals")
    ax[0].set_xlabel("residuals")
    ax[0].set_ylabel("count")


    plt.scatter(x=df_target["log existing followers"], y=results.resid, alpha=0.5, s=10)
    plt.xlabel("log existing followers")
    plt.ylabel("residuals")
    plt.title("Residuals vs log existing followers")

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


def plot_tsne(df, embeddings, target, subset=None):
    if subset:
        indices = df[df['tweet type'] == subset].index.tolist()
    filtered_df = df.iloc[indices] if subset else df
    filtered_embeddings = embeddings[indices] if subset else embeddings

    model = TSNE(n_components=2, perplexity=5, init='random', n_iter=1000, learning_rate=100)
    tsne = model.fit_transform(filtered_embeddings).T

    plt.gca().set_facecolor("black")
    scatter = plt.scatter(x=tsne[0], y=tsne[1], s=40, alpha=0.9, cmap="magma", c=filtered_df[target], edgecolors="white", linewidths=0.2)
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(filtered_df["Tweet text"].to_list()[sel.index])
        sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=1)
    plt.show()


months = ["may", "june", "july", "august", "september"]
file_arr = [f"data/processed/{month}_2023.xlsx" for month in months]

df = load_data(file_arr, follower_threshold=None, duration_threshold=None, exclude_multitags=False, exclude_selected_accs=False)
df = df.reset_index()
embedding_paths = [f"data/embeddings/{month}_2023_embeddings.pickle" for month in months]
embeddings = data_utils.load_embeddings(embedding_paths)

# plot_histogram(df)
# plot_scatter(df, "log impressions", "user profile clicks ratio")
plot_regression(df[df["tweet type"] == "Tweet"], "log existing followers", "log impressions")
ols(df, "Tweet")
# plot_tsne(df, embeddings, "user profile clicks ratio", subset="Tweet")

