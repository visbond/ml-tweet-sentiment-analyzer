# Tweet sentiment analyzer
This is a basic sentiment analyzer that uses relatively simple ML models to classify tweets on airlines as positive, neutral, or negative. It uses a data bank of 5000 tweets which is 80:20 split for training and testing. The text is cleaned and preprocessed (RegEx, stopword removal, Lemmatizing) before it is vectorized and passed through the training classifier. A variety of simple ML approaches are used, including Bayesian ones (Bernoulli Naive Bayes and Multinomial Naive Bayes), and others like SVM, Decision Trees, and a rudimentary neural net (basic Sklearn multilayer perceptron). The final version uses an ensemble voting classifier to get up to 80% accuracy.

This was made for an assignment in the Artificial Intelligence course at the University of New South Wales (UNSW). More details are in the assignment specification PDF. The subject code has not been used in plaintext in this readme or the repo title to prevent current students from finding it in case the instructor reuses this assignment. The students might get tempted to plagiarize from it, and there are heavy penalties for that.

Unfortunately the tweet databank is not allowed to be shared as it breaches Twitter's terms of service (the assignment specification states this clearly). It is a simple TSV file with three columns: a serial number in the first column, the actual tweet text in the second, and the rating ('positive', 'negative', 'neutral') in the third column. The tweets often contains URLs, special characters, and junk text, so cleaning was necessary.

How to run the file is given in the assignment specification (Assignment2.pdf); however the commands won't work without the tweet dataset.

A report that answers some theoretical and practical questions asked in the assignment is also included. It gives some insight into this class of NLP problems.

As this project was made in the early days of learning ML, the final voting classifier isn't the best in terms of design. It breaks ties by favoring the most likely contender, which would actually introduce bias in the model and make it prone to overfit the training data. It is much more fair to break ties randomly. This makes a difference of ~1% in accuracy when tested. However the code is being left as is, as an authentic representation of my knowledge at that point in time.

A much more sophisticated tweet sentiment analyser was made for the Machine Learning course in the next term, which would be uploaded here in due course. That achieves ~84% accuracy, and uses bidirectional LSTMs. It does not just predict the rating (positive or negative) but also what category a review belongs to (out of five, such as airlines, hotels etc).
