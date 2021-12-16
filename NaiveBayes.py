#Alex Szabo
#ITP 499 Fall 2021
#Final Project

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data into dataframe
review_data = pd.read_csv('amazon_reviews_us_Kitchen_v1_00.tsv'
                          , sep = '\t', on_bad_lines='skip')

review_data = pd.DataFrame(data=review_data
                           , columns=['marketplace', 'customer_id',	'review_id'
                                      , 'product_id', 'product_parent', 'product_title', 'product_category'
                                      , 'star_rating',	'helpful_votes', 'total_votes',	'vine'
                                      , 'verified_purchase', 'review_headline', 'review_body', 'review_date'])

# First 10 rows
pd.set_option("display.max_columns", None)
print(review_data.head(10))
# ----------------------------------------------------------------------------------------------
# Keep only review body and rating

reviews = review_data[['review_body', 'star_rating']]
print(reviews.head(10))
print('Total number of reviews:', str(len(reviews)))
# ----------------------------------------------------------------------------------------------
# Create positive and negative dataframes
# ----------------------------------------------------------------------------------------------
review_data_pos = reviews.loc[review_data['star_rating'] > 3]
review_data_neg = reviews.loc[review_data['star_rating'] < 3]

# ----------------------------------------------------------------------------------------------
# Randomly select 100,000 positive reviews and 100,000 negative reviews. Combine them into one dataframe.
# ----------------------------------------------------------------------------------------------
pos_sample = review_data_pos.sample(n=100000)
neg_sample = review_data_neg.sample(n=100000)

frames = [pos_sample, neg_sample]
combined_df = pd.concat(frames)
combined_df['sentiment'] = ""

combined_df.loc[combined_df['star_rating'] == 5.0, 'sentiment'] = 1
combined_df.loc[combined_df['star_rating'] == 4.0, 'sentiment'] = 1
combined_df.loc[combined_df['star_rating'] == 2.0, 'sentiment'] = 0
combined_df.loc[combined_df['star_rating'] == 1.0, 'sentiment'] = 0

# ----------------------------------------------------------------------------------------------
# Calculate the average length of all the reviews present in the data frame.
# ----------------------------------------------------------------------------------------------
avg_word_len = 0
for text in combined_df['review_body']:
    if not(pd.isna(text)):
        avg_word_len += len(text.split())
mean_length = avg_word_len/200000
print("Average length:", mean_length)
print(combined_df.head(3))
# ----------------------------------------------------------------------------------------------
# Data Cleaning
# ----------------------------------------------------------------------------------------------
#a. Convert all the reviews to lowercase
combined_df['review_body'] = combined_df['review_body'].str.lower()
#b. Remove URLs from reviews
combined_df['review_body'] = combined_df['review_body'].str.replace(r'http\S+',r'', regex = True).str.strip()

#c. remove non-alphabetical characters
combined_df['review_body'] = combined_df['review_body'].str.replace(r',',r' ', regex = True)

combined_df['review_body'] = combined_df['review_body'].str.replace(r'[^a-zA-Z\s]',r'', regex = True)

#d. Remove extra spaces between words
combined_df['review_body'] = combined_df['review_body'].str.replace(r'\s+',r' ', regex = True)

#e. perform contractions on reviews
# import contractions

# def contractionfunction(s):
#     return ' '.join([contractions.fix(word) for word in s.split()])

# combined_df['review_body'] = combined_df['review_body'].astype(str).apply(contractionfunction)

# ----------------------------------------------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords

#Import stop word set
stop_words = set(stopwords.words('english'))
print("Before preprocessing:\n", combined_df.head())

# Remove stop words
def stop_word_remover(x):
    return " ".join([word for word in x.split() if word.lower() not in stop_words])

combined_df['review_body'] = combined_df['review_body'].astype(str).apply(stop_word_remover)
print("After stop removal:\n\n", combined_df['review_body'])

# whitespace_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Lemmatize words and apply to dataset
def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

combined_df['review_body'] = combined_df['review_body'].apply(lambda x: lemmatization(x))

print('After lemmatization:\n\n', combined_df['review_body'])
# ----------------------------------------------------------------------------------------------
# Calculate the new average word length of all the reviews present in the data frame
# ----------------------------------------------------------------------------------------------

avg_word_len = 0
for text in combined_df['review_body']:
    if not(pd.isna(text)):
        avg_word_len += len(text.split())
mean_length = avg_word_len/200000
print("Average length:", mean_length)
print(combined_df.head(3)) # Result : average length has gone down substantially
# ----------------------------------------------------------------------------------------------
#10. Partition dataset into train and test, using sklearn train_test_split
# ----------------------------------------------------------------------------------------------

X = combined_df['review_body']
y = combined_df['sentiment']
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2021)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# ----------------------------------------------------------------------------------------------
# Extract the tfidf features from train and test dataset using sklearn TfidfVectorizer.
# ----------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train) # Fit_transform the train data
X_test = vectorizer.transform(X_test)   # transform test data
# ----------------------------------------------------------------------------------------------
# Train multinomial naive Bayes, and accuracy score for train
# ----------------------------------------------------------------------------------------------
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Instantiate
naive_bayes = MultinomialNB()

# Fit model
naive_bayes.fit(X_train, y_train)

# Predict on train data
y_pred_train = naive_bayes.predict(X_train)

# Train accuracy score:
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train accuracy:", accuracy_train)
# ----------------------------------------------------------------------------------------------
# Accuracy score on test data:
# ----------------------------------------------------------------------------------------------
# Predict on test data
y_pred_test = naive_bayes.predict(X_test)

#accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test accuracy:", accuracy_test)
# ----------------------------------------------------------------------------------------------
# Plot confusion matrix
# ----------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
print(confusion_matrix(y_test, y_pred_test))
plot_confusion_matrix(naive_bayes, X_test, y_test)
plt.show()
# ----------------------------------------------------------------------------------------------
# Print the review body, actual sentiment and the predicted sentiment of any inputted review.
# ----------------------------------------------------------------------------------------------
sample = str(input('Which review would you like to predict?: '))
sample = lemmatization(sample)
sample = vectorizer.transform([sample])
print(naive_bayes.predict(sample))