# 9414 Assignment 2
# own sentiment.py , the main classifier is in the statClassifier() class (for statistical classifier)
# gives 80% accuracy on given data set (4000 training + 1000 test tweets)

import pandas #to load the tsv
import re #to remove URLs and form tokeniser for CountVectoriser
import random
from sys import argv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter # used for mode calculation and bias
# from statistics import mode #fails in case of ties
# from nltk.corpus import wordnet
# from nltk.stem import WordNetLemmatizer # didn't improve accuracy, maybe due to informal, ungrammatical language and small example lengths
# from nltk.tokenize import word_tokenize # didn't lead to meaningful improvement, probably due to same reasons as above
# from nltk import pos_tag # ditto as above
# from collections import defaultdict # used for mapping pos_tag output to WordNet, for lemmatizing

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
so this just removes single letter/digits, to have words as two-letter units.
This also does stopword removal and stemming'''
def cv_tokenizer(tweet): 
    
    pattern = re.compile(r'\s\w\s') # catches a single letter or number with a space on each side. 
                                    # We are required to have words of min length two
    tweet = re.sub(pattern, ' ', tweet)  # remove above isolated single letters with spaces on each side,
                                                # replace with single whitespace
    tokens = tweet.split() # is faster than nltk word tokenizer, and similar results on this dataset
    stemmer = PorterStemmer()
    stopwordset = set(stopwords.words('english'))
    stems = []
    for token in tokens:
        if token not in stopwordset:
            stems.append(stemmer.stem(token))
    return stems
#end def cv_tokenizer()

'''lemmatized version of above tokenizer. Negligible gain in performance,
but slower speed (as wordnet needs to be loaded)
Also led to some warnings, so risky for automarked assignment. 
This function is left here for reference but not being used.
Comment the other cv_tokenizer() and uncomment this if want to use.
Will also need to uncomment the relevant imports'''
'''
def cv_tokenizer(tweet): #Q4SPECIFIC two lines changed inside for stemming
    
    pattern = re.compile(r'\s\w\s') # catches a single letter or number with a space on each side. 
                                    # We are required to have words of min length two
    tweet = re.sub(pattern, ' ', tweet)  # remove above isolated single letters with spaces on each side,
                                                # replace with single whitespace
    
    tokens = word_tokenize(tweet)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tweet = []
    
    # note on lemmatizing: a lemmatizer converts words to synonyms based on their meaning, 
    # unlike a stemmer, which only uses letter syntax. Supplying a part of speech tag to the lemmatizer
    # improves performance. By default it assumes the word sent to it is a noun, but we can change this.
    # e.g. by default "better" is lemmatized to "better", but when we specify it is an adjective, it is
    # lemmatized to "good". The issue is that the pos tags returned by nltk.pos_tag() are not all supported
    # by the WordNet lemmatizer, and they have different names as well. The following dictionary does the
    # mapping for some common parts of speech, and returns noun by default.
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['S'] = wordnet.ADJ  # satellite adjective
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV
    
    for word, tag in pos_tag(tokens):
        lemma = lemmatizer.lemmatize(word, tag_map[tag[0]]) # only checks first letter of tag to determine part of speech
        lemmatized_tweet.append(lemma)
        
    stemmer = PorterStemmer()
    stopwordset = set(stopwords.words('english')) 
    final_words = []
    for word in lemmatized_tweet:
        if word not in stopwordset:
            final_words.append(stemmer.stem(word))
            # final_words.append(word)
    return final_words
#end def cv_tokenizer()
'''

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

'''ensemble implementation for classification; uses multiple classifiers, and calculates
the most common class label (incorporates tiebreaking) to classify examples'''
class statClassifier:
    def __init__ (self):
        self.NUM_CLASSES = 3 # we currently have 3 classes; this is for future flexibility for use in a different environment
        self.dt_clf = tree.DecisionTreeClassifier(min_samples_leaf=.01, criterion='entropy', random_state=0)
        self.mnb_clf = MultinomialNB()
        self.bnb_clf = BernoulliNB()
        self.lr_clf = LogisticRegression(solver='lbfgs', multi_class = 'auto')
        self.lsvc_clf = LinearSVC()
    #end def
    
    '''returns the most common class label'''
    def __find_bias__(self, labels):
        label_count_dic = Counter(labels)
        label, count = label_count_dic.most_common(1)[0] #find the single most common class, if there is a tie,
            #must take arbitrarily whichever is first, since at this point we have only the training set
            # and don't have further information (proper tiebreaking is done with the testing set in the
            #custom_mode function)
        # print("DEBUG, in find_bias, most common training label and count is:", label, count)
        return label
    #end def
    
    '''finds the mode of the given list, with tiebreaking'''
    def __custom_mode__(self, labels):
        label_count_dic = Counter(labels)        
        top2_counts = label_count_dic.most_common(2)
        # print("DEBUG in custom_mode, top2_counts is", top2_counts)
        if len(top2_counts) == 1: # all classifiers agree
            return top2_counts[0][0]
        else: #have at least two classes
            if top2_counts[0][1] == top2_counts[1][1]: # we have a tie in number of counts of those classes
                if (top2_counts[0][0] == self.bias) or (top2_counts[1][0] == self.bias): 
                # if either of those classes is the same as our bias, return it
                    return self.bias
            
        # in all other cases return the first class, since we have no more reliable information to break the tie
        return top2_counts[0][0]
        #end def
        
    def train(self, bags_of_words, labels):
        self.dt_model = self.dt_clf.fit(bags_of_words, labels)
        self.mnb_model = self.mnb_clf.fit(bags_of_words, labels)
        self.bnb_model = self.bnb_clf.fit(bags_of_words, labels)
        self.lr_clf = self.lr_clf.fit(bags_of_words, labels)
        self.lsvc_clf = self.lsvc_clf.fit(bags_of_words, labels)
        self.bias = self.__find_bias__(labels) # will hold the most common label of the class, used for breaking ties
    #end def
    
    def predict(self, bags_of_words):
        dt_predict = self.dt_model.predict(bags_of_words)
        mnb_predict = self.mnb_model.predict(bags_of_words)
        bnb_predict = self.bnb_model.predict(bags_of_words)
        lr_predict = self.lr_clf.predict(bags_of_words)
        lsvc_predict = self.lsvc_clf.predict(bags_of_words)
        predictions = []
        for i in range (len(dt_predict)):
            predictions.append(self.__custom_mode__([dt_predict[i], mnb_predict[i],
                bnb_predict[i], lr_predict[i], lsvc_predict[i]]))
        return predictions
#end class

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
# count = CountVectorizer (tokenizer=cv_tokenizer, lowercase = True, max_features = 1000) 
count = CountVectorizer (tokenizer=cv_tokenizer, lowercase = True, max_features = 500) 

# fit with training data
X_train_bag_of_words = count.fit_transform(X_train) # essentially, creates a feature vector, with word counts as values

custom_clf = statClassifier()
custom_clf.train(X_train_bag_of_words, y_train)

# load test set, in similar way as training set. Assuming the file only has instance number and tweet text columns, no sentiment
test_set = pandas.read_csv(test_file, sep = '\t', header = None)

test_instance_numbers = test_set[0].tolist() # first column of TSV
X_test = test_set[1].tolist() # second column
y_test = test_set[2].tolist() # third column, class labels

# remove URLs and special characters
for i in range (len(X_test)):
    X_test[i] = removeURLandSpecialChars(X_test[i])

# transform the test data into bag of words with count vectorizer object
X_test_bag_of_words = count.transform(X_test)

predicted_y = custom_clf.predict(X_test_bag_of_words)

#print columns side-by-side as required
print_columnwise(test_instance_numbers, predicted_y) 


##### diagnostic and reporting code

# def bucket_count (sentiment_list):
#     neucount = 0
#     negcount = 0
#     poscount = 0
#     for senti in sentiment_list:
#         if senti == 'negative':
#             negcount += 1
#         elif senti == 'positive':
#             poscount += 1
#         elif senti == 'neutral':
#             neucount += 1
#         else:
#             print("ERROR, unclassed value")
#             break
#     return (poscount, neucount, negcount)
# #end def

# counts = bucket_count (predicted_y) # result is tuple in order positive, neutral, negative counts
# print (counts)
# print(classification_report(y_test, predicted_y))

