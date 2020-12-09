# Tweet sentiment analyzer
This is a basic sentiment analyzer that uses relatively simple ML models to classify tweets on airlines as positive, neutral, or negative. It uses a data bank of 5000 tweets which is 80:20 split for training and testing. The text is cleaned and preprocessed (RegEx, stopword removal, Lemmatizing) before it is vectorized and passed through the training classifier. A variety of simple ML approaches are used, including Bayesian ones (Bernoulli Naive Bayes and Multinomial Naive Bayes), and others like SVM, Decision Trees, and a rudimentary neural net (basic Sklearn multilayer perceptron). The final version uses an ensemble voting classifier to get up to 80% accuracy.

This was made for an assignment in the Artificial Intelligence course at the University of New South Wales (UNSW). More details are in the assignment specification. The subject code has not been used in plaintext in this readme or the repo title to prevent current students from finding it in case the instructor reuses this assignment. They might be tempted to plagiarize from it, and there are heavy penalties for that.

Unfortunately the tweet databank is not allowed to be shared as it breaches Twitter's terms of service (the assignment specification states this clearly). It is a simple TSV file with a serial number in the first column, the actual tweet text in the second, and the rating ('positive', 'negative', 'neutral') in the third and last column. The tweets often contains URLs, special characters, and junk text, so cleaning was necessary.

A much more sophisticated tweet sentiment analyser was made for the Machine Learning course in the next term, which would be uploaded here in due course. That achieves ~84% accuracy, and uses bidirectional LSTMs. It does not just predict the rating (positive or negative) but also what category a review belongs to (out of five, such as airlines, hotels etc).
