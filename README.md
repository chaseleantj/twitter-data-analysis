## Analysis of Twitter data

I have a Twitter (now X) account @chaseleantj, and I made 3974 posts from 12 May 2023 to 30 September 2023.

This personal project aims to use data analysis and machine learning to discover interesting insights from the data I've collected.

The 3974 posts are grouped into 5 categories:

| Category        | Count |
|-----------------|-------|
| Reply           | 2964  |
| Thread content  | 722   |
| Tweet           | 219   |
| Quote           | 41    |
| Community       | 28    |

### Key findings

<ul>
<li>My 219 tweets gained a minimum, median and maximum view count of 219, 56K and 7.8M views respectively, log-normally distributed with a standard deviation of 57K.</li>
<li>Unsurprisingly, the more views a post gets, the more engagement it receives. But as posts get more views, the ratio of likes, replies and user profile clicks to view count decreases (correlation -0.46, -0.32 and -0.17) respectively. This can be interpreted as popular posts reaching a wider, and hence, less targeted audience.</li>
<li>Tweets featuring tutorials received the highest amount of engagement and more than 5x the number of views. People love tutorials.</li>
<li>Follower milestone celebration tweets had a 10x higher user profile click ratio to impressions vs regular tweets. However, they usually have 5x lower impressions.</li>
<li>By simple linear regression with log-log axes, a 1% increase in follower count increases the mean view count of a tweet by 0.55%, with $R^{2}=0.263, F(1, 217)=77.54, p=4.19\times 10^{-16}$.</li>
<li>Interestingly, a 1% increase in follower count also increases the mean view count of a reply by 0.39%, with $R^{2}=0.155, F(1, 2962)=541.6, p=3.64\times 10^{-110}$.</li>
<li>The linear relationship between view count and follower count is weak. This means that larger accounts do indeed have an advantage, but smaller accounts can have viral posts too - provided their content is good.</li>
<li></li>
<li>A deep neural network was trained to classify an unknown post into one of the 5 categories using the post's text content alone, via embeddings with OpenAI's Ada-002. It achieved an accuracy of 95%. However, this is not superior to SVMs and Logistic Regression classifiers, likely due to the small dataset (relative to the number of features and parameters).</li>
</ul>
