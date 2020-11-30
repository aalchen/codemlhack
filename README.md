# codemlhack
CodeML AI/ML Hackathon - Nov 2020

These are entries to the CodeML Hackathon, challenge #1 and challenge #4.
Prior to attending this hackathon, I had no experience working with AI/ML algorithms.

Challenge #1 - Clustering Analysis
Score tied for #1, ranked #9
https://www.kaggle.com/c/codeml-challenge1

DBScan, Gaussian Mixture, and K-means clustering algorithms were tested and explored, however determined to be ineffective for this use case.
Principle Component Analysis (PCA) was used to work with the multi-dimensionality of this data.
Ultimately, the Spectral Clustering algorithm was selected to determine the clustering of stars.

Challenge #4 - Sentiment Analysis
Scored #2, ranked #2
https://www.kaggle.com/c/codeml-challenge4

Sentiment Analysis required working with Natural Language Processing.
After inaccuracies with Naive Bayes algorithm and Random Forest Classifiers, Bernoulli Bayes Classifier was the optimal algorithm given the boolean nature of the data. 
Significant data wrangling was also required using regex, and manual adjustments were required to account for negations.
