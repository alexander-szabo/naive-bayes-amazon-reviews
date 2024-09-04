# naive-bayes-amazon-reviews
- The goal of this notebook is to create a Naïve Bayes NLP model for predicting positive or negative Amazon reviews.

- Naive Bayes is a type of Machine Learning algorithm based in statistics (Bayes' Theorem) which is convenient to use in classification problems for high-dimension data. It can be used to predict sentiment analysis of text-related data, where we assume each of the input features are conditionally independent of each other (the 'prior' distribution).

- In this notebook, I implement a Multinomial Naive Bayes model in order to predict the sentiment of Amazon reviews for common kitchen items -- whether they are 'positive' or 'negative' reviews. The model is accurate to ~87% of cases in the test sample; it correctly classifies 'positive' as positive and 'negative' as negative 87% of our test results.
