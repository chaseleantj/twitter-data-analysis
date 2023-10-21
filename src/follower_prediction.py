import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

followers = pd.read_csv("data/followers.xlsx")

# Create a new column, which is the difference between the current date and the first date
followers["date"] = pd.to_datetime(followers["date"])
followers["duration"] = (followers["date"] - followers["date"].iloc[0]).dt.days

# Make prediction line
t = np.arange(0, 270)
y = 0.347938 * (3.601 * t + 6.47031) ** 1.798561151079137
y_revised = 0.347938 * (4.651 * t - 38.692) ** 1.798561151079137

# Change t to a datetime object
t = pd.to_datetime(t, unit="D", origin="2023-05-12")

ax = plt.gca()

plt.plot(followers["date"], followers["followers"], linewidth=2, label="Actual follower count")
plt.plot(t, y, color="orange", linewidth=1, label="Predicted follower count", linestyle="dashed")
plt.plot(t, y_revised, color="red", linewidth=1, label="Revised predicted follower count", linestyle="dashed")

plt.title("Follower count over time")
plt.xlabel("Time")
plt.ylabel("Follower count")
plt.legend()
plt.grid(True)

plt.show()