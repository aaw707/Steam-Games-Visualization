# import the libraries
import numpy as np
import pandas as pd

# import the files
steam = pd.read_csv('steam.csv')
steam_description = pd.read_csv('steam_description_data.csv')
steamspy_tags = pd.read_csv('steamspy_tag_data.csv')

# ----------

# Prep data for Bubble Charts

# 1
# get tag counts

tag_counts = pd.DataFrame(steamspy_tags.sum(axis =0).sort_values(ascending = False)[1:]).reset_index()
tag_counts.rename(columns = {'index': 'Token',
                                   0: 'Counts'}, inplace = True)                               
tag_counts['Type'] = 'Tag'


# 2
# get category counts

# clean the string values into lists
steam['categories'] = steam['categories'].apply(lambda x: x.split(";")) 

# append all occurance of categories into a list
full_cats = []
for i in range(len(steam)):
  for cat in steam['categories'][i]:
    full_cats.append(cat)

# counts by category
cat_counts = pd.DataFrame(pd.Series(full_cats).value_counts(ascending = False)).reset_index()
cat_counts.rename(columns = {'index': 'Token',
                                   0: 'Counts'}, inplace = True)
cat_counts['Type'] = 'Category'
 

# 3
# get genre counts

# clean the string values into lists
steam['genres'] = steam['genres'].apply(lambda x: x.split(";"))

# append all occurance of genres into a list
full_genres = []
for i in range(len(steam)):
  for genre in steam['genres'][i]:
    full_genres.append(genre)

# counts by genre
genre_counts = pd.DataFrame(pd.Series(full_genres).value_counts(ascending = False)).reset_index()
genre_counts.rename(columns = {'index': 'Token',
                                     0: 'Counts'}, inplace = True)
genre_counts['Type'] = 'Genre'


# 4
# concat the 3 dataframes 

# in order to switch views on the dashboard, the data source of these three charts should be the same file
bubble_data = pd.concat([tag_counts, cat_counts, genre_counts], axis = 0).reset_index(drop = True)

# export
bubble_data.to_csv('bubble_data.csv')


# ----------

# 4
# get the most frequent words from the short description of top 100 games (by number of positive ratings)

# id of top 100 games (by number of positive ratings)
top_100_game_id = steam.sort_values(by = 'positive_ratings', ascending = False).head(100)['appid'].reset_index(drop = True)

# filter out descriptions of the top 100 games
short_des = pd.merge(top_100_game_id, steam_description, left_on = 'appid', right_on = 'steam_appid')['short_description']

# strip html
import re
short_des = short_des.apply(lambda x: re.sub('<[^<]+?>', '', x))
short_des = short_des.apply(lambda x: x.replace('&quot', '').replace('\r\n', ''))

# convert to lowercase
short_des = short_des.apply(lambda x: x.lower())
short_des

# add additional stop words, found from EDA
from sklearn.feature_extraction import text  
stop_words = text.ENGLISH_STOP_WORDS.union(
    {'new', 'game', 'play', 'players', 'player', 'person', 'set', 'open', 'way', 'don', 'year', 'gameplay', 'like', 'includes'})

# vectorize the words in short descriptions and find the top 50 frequent words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = stop_words, min_df = 3, max_features = 50)
cv.fit(short_des)
short_des_transformed = cv.transform(short_des)

# put the results in a dataframe
short_des_vectorized = pd.DataFrame(columns = cv.get_feature_names(), 
                                    data = short_des_transformed.toarray())

# count the occurance of the top 50 frequent words (tokens)
tokens = pd.DataFrame(short_des_vectorized.sum(axis = 0).sort_values(ascending = False)).reset_index()
tokens.rename(columns = {'index': 'token',
                               0: 'Counts'}, inplace = True)

# export
tokens.to_csv('short_description_top_words.csv')