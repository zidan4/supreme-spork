#import basic visualization libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#clean columns

from google.colab import drive
drive.mount('/content/drive')

!ls drive/'My Drive'/ML_Process_Data_Files/Section_6_EDA

#read in data
df_agg = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/Aggregated_Metrics_By_Video.csv',encoding='utf-8')
df_agg_country_sub = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv', encoding='utf-8')
df_ts = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/Video_Performance_Over_Time.csv', encoding='utf-8')
df_comments = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/All_Comments_Final.csv', encoding='utf-8')

#look at columns for each dataframe
print(df_agg.columns)
print(df_agg_country_sub.columns)
print(df_ts.columns)
print(df_comments.columns)

#the column headers have some extra non-ascii characters, we need to clean them up before we do our analysis
#this goes through each column and removes all the non-ascii characters

newcols =[x.encode("ascii", "ignore").decode('utf-8') for x in df_agg.columns]
df_agg.columns = newcols

df_agg.columns

#Using .describe() gives us basic descriptive statistics for all of our numeric data. From this we can see which variables we might want to explore more.
#We can see things like: which variables might have outliers, which might have skew, or which have a wide range of values
df_agg.describe()

"""## Single variable plots
After looking at the descriptive statistics, we may want to explore our numeric features more. To do this, we like to use single variable plots. These can help us understand the distributions of our data. If our data doesn't follow a normal distribution, we may want to make some transform it so it can be used by specific types of models like linear regresison.

Looking at these charts can also help us to evaluate if there are outliers present in our data.

First, we will look at histograms of our data. These help us to see skew as well as some outliers. I usually do this with every feature (if possible). Try more of these on your own!

Next, we will explore some box plots to see if they tell us additional information

Finally, we will explore some distribution plots for categorical variables.
"""

#you can easily see the distribuiton of data in a column with panda's built in .hist() method. Here is the distribution for likes. There is clearly one sample that has way more like sthan the others, this is something we could look into more

df_agg.Likes.hist(bins = 100)

#let's look at the same thing for average percentage viewed. This will have no outliers unlike the likes column. We are using matplotlib's hist function here instead of the integration through pandas.
plt.hist(df_agg['Average percentage viewed (%)'])

#this data has a significant amount of right skew, we may want to transform this data if we were planning to use linear regression.
df_agg['Impressions click-through rate (%)'].hist()

#From this boxplot, we can see that there are quite a few outliers from our normal data. What does this tell us about the nature of likes on videos? Perhaps we shouldn't be trusting averages if something can skew so high.
plt.boxplot(df_agg['Likes'])

#this plot looks a lot more normal. We can se emost of the videos are viewed between 25-45% of the way through. We still have some outliers, what may be special or different about those?
plt.boxplot(df_agg['Average percentage viewed (%)'])
#

#with click through rate, we also see quite a few high outliers versus the median of / interquartile range.
plt.boxplot(df_agg['Impressions click-through rate (%)'])

#To show how to do this with categorical data, let's make a categorical column.
#We can take our revenue data and make it into different categories that are relevant to me. Usually I'm interested if videos have made less than $100,
#between $100-1000 and over $1000.

#Let's quickly "engineer these categories for ourself"

#make bins from 0-100, 100-1000, and greater than 1000
bins = pd.IntervalIndex.from_tuples([(0, 100), (100, 1000), (1000,float("inf"))])
df_agg['rev_buckets'] = pd.cut(df_agg['Your estimated revenue (USD)'],bins)

#get count of number of videos by reveune bucket
rev_values = df_agg['rev_buckets'].value_counts()
rev_values.plot.bar()

#We can get a sense of the balance of categorical variables using a bar chart like this.

"""## Relationships and Multi-Variable Plots
A big part of exploratory data analysis is seeing how mutliple variables are related. We can use multiple different types of plots to easily see these relationships. By understanding these relationships we can start to udnerstand which features may be releated or can serve to predict others. In my opinion, this is where buisiness value starts to emerge.

In this part we will explore:

1) Scatter plots

2) Correlation Matrices

3) Pivot Tables

4) Bar Charts

5) Line Charts

### Scatter Plots

We use these to see if there is a relationship between two datapoints.

Let's take a look  at a few variables and see if they may be correlated. We will explore if the average percentage viewed of the video is related to the cost per milli on the video (the amount youtube makes for 1000 views)
"""

#create plot with matplotlib
plt.scatter(df_agg['Average percentage viewed (%)'] ,df_agg['CPM (USD)'])

#let's do the same thing with seaborn so we can see a trendline. This will be easier for us to build multiple of these
sns.regplot(x='Average percentage viewed (%)',y='CPM (USD)', data = df_agg)

#as we can see there is a very little correlation between the two variables

#Let's try another. How do variables RPM and CPM match up. RPM is how much the youtuber makes for an add. Again CPM is how much youtube sells the ad for.
sns.regplot(x='RPM (USD)',y='CPM (USD)', data = df_agg)

# Here we can see a lot stronger relationship between the two rates.
# I should probably look into that one value where YouTube is making a massive margin on my video. These are the types of insights you can find in these graphs

"""### Correlation Matrices

Scatter plots are great for comparing two variables. Often we have many different variables that we want to see relationships between.


In this case we would use a correlation matrix. Let's do a correlation matrix for all of our value in the dataset.

"""

#first, we get the correlations between our datapoints. We can see all the relationships that will be polotted via the pearson correlation coefficient
corr = df_agg.corr()

sns.heatmap(corr)

#this looks pretty awful, let's improve it's look

#A better example (formatting used in below chart) - https://seaborn.pydata.org/examples/many_pairwise_correlations.html

sns.set_theme(style="white")

# Compute the correlation matrix
corr = df_agg.corr()

# Generate a mask for the upper triangle (otherwise this looks like the square we had above and is redundant)
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Generate a custom diverging colormap (choose colors here)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"fontsize":8})

#obviously many of thes variables are HIGHLY correlated. Something we may want to explore is why Average percentage viewed is negatively related to RPM

"""###Pivot Tables
Sometimes we want to cut the data and compare how variables stack up. For example we may want to see which countries have the highest view duration. We can do this easily with pivot tables.

We will use the df_agg_country_sub dataset to explore this.

"""

#pivot table to explore values. A basic pivot table takes the average of these different categories we choose.

pd.pivot_table(df_agg_country_sub, index = 'Country Code', values = 'Average View Percentage')

#this time, let's compare average view percentage by country code and subscriber status.
pd.pivot_table(df_agg_country_sub, index = 'Country Code', columns = 'Is Subscribed',values = 'Average View Percentage')

#we could plot this to see which countries have the biggest difference in subscribed watch time for subscribed vs not subscribed viewers

#Let's plot a slightly simpler graph. Let's just look at if subscribers or non-subsribers watch my videos for longer
pd.pivot_table(df_agg_country_sub, index = 'Is Subscribed', values = 'Average View Percentage').plot.bar()

#it appears that subscribers to my channel watch my videos for longer than non-subscribers. This is another way that we can use a bar chart to compare multiple features

"""### Line Charts
Often, we want to see how variables change over time. Line charts are great for this. Time is an important feature in many models!

We will use the df_ts dataset for this analys.
"""

#First we need to make sure our date field is in the date time format. We do this by converting our string to datetime
df_ts['Date'] = pd.to_datetime(df_ts['Date'])

#let's look at user subscriptions removed and see if there is any trend there. We will also compare this with user likes removed to see if we can find anything interesting

#first, we have to aggregate these by video. We do this with a pivot table.
rm_x_date = pd.pivot_table(df_ts, index='Date',values = 'User Subscriptions Removed', aggfunc ='sum').reset_index()

#next we visualize this data with seaborn
sns.lineplot(data=rm_x_date,x='Date', y='User Subscriptions Removed')

#now let's compare this with two other related metrics, video likes removed and dislikes

likes_rm_date = pd.pivot_table(df_ts, index='Date',values='Video Likes Removed', aggfunc = 'sum').reset_index()
dislikes_date = pd.pivot_table(df_ts, index='Date',values='Video Dislikes Added', aggfunc = 'sum').reset_index()

sns.lineplot(data=rm_x_date,x='Date', y='User Subscriptions Removed')
sns.lineplot(data=likes_rm_date,x='Date', y='Video Likes Removed')
sns.lineplot(data=dislikes_date,x='Date', y='Video Dislikes Added')

#This was a bit messy, let's compare it over months instead of days.
df_ts['Month_Year'] = df_ts['Date'].dt.to_period('M')

#create new pivot tables
rm_x_date = pd.pivot_table(df_ts, index='Month_Year',values = 'User Subscriptions Removed', aggfunc ='sum').reset_index()
likes_rm_date = pd.pivot_table(df_ts, index='Month_Year',values='Video Likes Removed', aggfunc = 'sum').reset_index()
dislikes_date = pd.pivot_table(df_ts, index='Month_Year',values='Video Dislikes Added', aggfunc = 'sum').reset_index()

#create 3 separate line plots with pandas built in visualization tool
rm_x_date.plot(x='Month_Year',y='User Subscriptions Removed')
likes_rm_date.plot(x='Month_Year',y='Video Likes Removed')
dislikes_date.plot(x='Month_Year',y='Video Dislikes Added')

#This was a bit messy, let's compare it over months instead of days. To get everything on one graph, we need to make it so every video has the same day.
#Just different months and years
df_ts['Month_Year'] = df_ts['Date'].apply(lambda x: x.replace(day=1))

rm_x_date = pd.pivot_table(df_ts, index='Month_Year',values = 'User Subscriptions Removed', aggfunc ='sum').reset_index()
likes_rm_date = pd.pivot_table(df_ts, index='Month_Year',values='Video Likes Removed', aggfunc = 'sum').reset_index()
dislikes_date = pd.pivot_table(df_ts, index='Month_Year',values='Video Dislikes Added', aggfunc = 'sum').reset_index()

#create 3 line plots with seaborn all on one graph
sns.lineplot(data=rm_x_date,x='Month_Year', y='User Subscriptions Removed', label ='Subs Removed')
sns.lineplot(data=likes_rm_date,x='Month_Year', y='Video Likes Removed', label = 'Likes Removed')
sns.lineplot(data=dislikes_date,x='Month_Year', y='Video Dislikes Added', label = 'Dislikes')

#it seems like there is a big spike in unsubscribes and negative comments during this period. We should explore this more.
#maybe we should divide these all by views to determine if the spike is only related to a spike in viewership as well.
#or maybe a single video caused a lot of negative things this period

#Let's normalize this by views

df_ts['Month_Year'] = df_ts['Date'].apply(lambda x: x.replace(day=1))

monthly_views = pd.pivot_table(df_ts, index = 'Month_Year', values = 'Views', aggfunc = 'sum')
rm_x_date = pd.pivot_table(df_ts, index='Month_Year',values = 'User Subscriptions Removed', aggfunc ='sum') / monthly_views.values
likes_rm_date = pd.pivot_table(df_ts, index='Month_Year',values='Video Likes Removed', aggfunc = 'sum') / monthly_views.values
dislikes_date = pd.pivot_table(df_ts, index='Month_Year',values='Video Dislikes Added', aggfunc = 'sum') / monthly_views.values

#create 3 line plots with seaborn all on one graph
sns.lineplot(data=rm_x_date,x='Month_Year', y='User Subscriptions Removed', label ='Subs Removed')
sns.lineplot(data=likes_rm_date,x='Month_Year', y='Video Likes Removed', label = 'Likes Removed')
sns.lineplot(data=dislikes_date,x='Month_Year', y='Video Dislikes Added', label = 'Dislikes')