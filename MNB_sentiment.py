# 9414 Assignment 2, MNB_sentiment.py
# takes train and test files via commandline, makes predictions using Multinomial Naive Bayes classifier
# note that as per comment @278 in Piazza, the professor has clearly said:
# "Don't convert to lower case in the standard models."
# so using lowercase = False argument in CountVectorizer

import pandas #to load the tsv
import re #to remove URLs and form tokeniser for CountVectoriser
import random
from sys import argv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

'''remove URLs and unwanted characters from passed text, using regular expressions.
Note that this also removes punctuation. Otherwise 'flight', 'flight,','flight.' and 'flight-' would be counted
as four different words, even though they refer to the same thing.'''
def removeURLandSpecialChars(tweet):
    pattern_URL = re.compile(r'https?://(www\.)?(\w+)(\.\w+)(/\w+)?')
    # assumes URL begins with http or https, then may or may not have www., then site name, then .TLD, then may nor may not have webpage name
    # random note: BNB classification for tweet 4995 changes from negative to positive if this function is applied, is negative otherwise
    
    pattern_chars = re.compile(r'[^a-zA-Z0-9_@#$% ]') 
    # will match anything that is not an alphanumeric character or _ @ # $ % or space.
    # Note that ^ usually indicates the beginning of a string, but inside a character set [], it denotes 'not'
    # Note there is a space character at the end of the character set, is necessary else space is also removed and output comes all concatenated
    
    tweet_noURL = re.sub(pattern_URL, '', tweet)
    result_tweet = re.sub(pattern_chars, '', tweet_noURL) #another option is to use a ' ' as the second argument here (instead of specifying space in re pattern above),
                                                        #but that would replace special characters with spaces, which we don't want if it is in the middle of a string
    return result_tweet #for tokenizer, return list of words
#end def removeURLs()

'''tokenizer for CountVectorizer(). Since special chars and URLs have already been removed, 
so this just removes single letter/digits, since we are required to have words as min two-letter units'''
def cv_tokenizer(tweet):
    pattern = re.compile(r'\s\w\s') # catches a single letter or number with a space on each side. 
                                    # We are required to have words of min length two
    result_tweet = re.sub(pattern, ' ', tweet)  # remove above isolated single letters with spaces on each side,
                                                # replace with single whitespace
    return result_tweet.split() #for tokenizer, return list of words
#end def cv_tokenizer()

'''Prints elements of two lists side by side, as required for final output
Has some additional parameters which were used for internal testing and reporting, not being used in final output'''
def print_columnwise (list1, list2, header1 = None, header2 = None):
    # size = list1.size()<list2.size()?list1.size():list2.size() # surprised to know Python doesn't have ternary operator
    if len(list1) <= len(list2):
        size = len(list1)
    else:
        size = len(list2)
    if header1 is not None:
        print(header1, header2)
    for i in range (size):
        print(list1[i], list2[i])
#end def


########################
### main starts here ###
########################

assert len(argv) >= 3 #to avoid more serious errors later if try to open non-existent files
train_file = argv[1]
test_file = argv[2]

train_set = pandas.read_csv(train_file, sep = '\t', header = None) #saying header = 0 actually takes first row as header
# note that now columns 0, 1, and 2 respectively hold the instance number, tweet text, and sentiment
# rename them to reduce chances of confusion/error, which can arise when handling purely numerical column names
train_set.columns = ['instance_number', 'tweet', 'sentiment']

instance_list = train_set['instance_number'].tolist()
tweet_list = train_set['tweet'].tolist()
senti_list = train_set['sentiment'].tolist()

# remove URLs and special characters
for i in range (len(tweet_list)):
    tweet_list[i] = removeURLandSpecialChars(tweet_list[i])

X_train = tweet_list
y_train = senti_list

# create count vectorizer object
count = CountVectorizer (tokenizer=cv_tokenizer, lowercase = False) #we need to call our custom tokenizer because the default one removes @
# lowercase is set to False as per comment by Prof in @278 on Piazza

# fit with training data
X_train_bag_of_words = count.fit_transform(X_train) # essentially, creates a feature vector, with word counts as values

clf = MultinomialNB()
model = clf.fit(X_train_bag_of_words, y_train)

# load test set, in similar way as training set. Assuming the file only has instance number and tweet text columns, no sentiment
test_set = pandas.read_csv(test_file, sep = '\t', header = None)

test_instance_numbers = test_set[0].tolist() # first column of TSV
X_test = test_set[1].tolist() # second column

# remove URLs and special characters
for i in range (len(X_test)):
    X_test[i] = removeURLandSpecialChars(X_test[i])

# transform the test data into bag of words with count vectorizer object
X_test_bag_of_words = count.transform(X_test)

# predict using MNB model
predicted_y = model.predict(X_test_bag_of_words)

#print predictions and true values side-by-side
print_columnwise(test_instance_numbers, predicted_y) 
 
