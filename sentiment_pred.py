import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def read_file(filepath):
    """
    Reads text file in and output list of tokenized reviews
    :param filepath: filepath to text file
    :return: list of tokenized reviews
    """
    with open(filepath) as f:
        reviews = f.read()
        reviews = reviews.split('\n')
        tokenized = map(lambda x: x.split(), reviews)
    return list(tokenized)


def split_vals(reviews, perc):
    """
    Splits data into training and test set
    :param reviews: List of tokenized reviews
    :param perc: percent of data to go into test set
    :return: Data splitted into train and test
    """
    n = int(len(reviews)*perc)
    return reviews[:n], reviews[n:]


def generate_w2v(tokens):
    """
    Given a set of tokenized reviews, trains word2vec model.
    :param tokens: Corpus of reviews, broken down into tokens for each review.
    :return: Trained Word2Vec model.
    """
    w2v_model = Word2Vec(size=100, window=10, min_count=5, workers=8,
                         alpha=0.025, min_alpha=0.025)
    w2v_model.build_vocab(tokens)
    for epoch in range(10):
        print "Iteration", epoch
        w2v_model.train(tokens, total_examples=w2v_model.corpus_count,
                        epochs=w2v_model.iter)
        w2v_model.alpha -= 0.001  # decrease the learning rate
        w2v_model.min_alpha = w2v_model.alpha
    return w2v_model


def get_non_stopwords(tokens, stopwords):
    """Returns a list of non-stopwords"""
    return [x for x in tokens if x not in stopwords]


def sentence_features(review_tokens, w2v_model, emb_size=100):
    """
    Turn review into average of the word embeddings
    :param review_tokens: Tokenized review
    :param w2v_model: Trained Word2Vec model object
    :param emb_size: Size of embedding
    :return: Average of word embeddings in the review
    """
    words = get_non_stopwords(review_tokens, stopWords)
    words = [w for w in words if w.isalpha() and w in w2v_model]
    if len(words) == 0:
        return np.zeros(emb_size)
    M = np.array([w2v_model[w] for w in words])
    return M.mean(axis=0)


def generate_XY(w2v_model, pos_data, neg_data):
    """Generated X and y variables from reviews data to input into model"""
    X = pd.DataFrame(list(map(lambda x: sentence_features(x, w2v_model), pos_data + neg_data)))
    y = [1] * len(pos_data) + [0] * len(neg_data)
    return X, y


if __name__ == '__main__':
    # Read in text files
    reviews_pos = read_file('positive_reviews.txt')
    reviews_neg = read_file('negative_reviews.txt')
    unsup = read_file('unsupervised_reviews.txt')
    # Split into training/test set
    pos_test, pos_train = split_vals(reviews_pos, .20)
    neg_test, neg_train = split_vals(reviews_neg, .20)
    # Aggregate all training reviews to train word2vec
    all_reviews = pos_train + neg_train + unsup
    # Train Word2Vec
    print('Training Word2Vec...')
    w2v = generate_w2v(all_reviews)
    # Load Stop Words
    stopWords = set(stopwords.words('english'))
    # Generate features for model
    X_train, y_train = generate_XY(w2v, pos_train, neg_train)
    X_test, y_test = generate_XY(w2v, pos_test, neg_test)
    # Run various models
    lin_svm = svm.LinearSVC()
    lin_svm.fit(X_train, y_train)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    logisticRegr = LogisticRegression(C=1.2)
    logisticRegr.fit(X_train, y_train)
    # Output accuracy
    print('SVM Accuracy: ', lin_svm.score(X_test, y_test))
    print('Naive Bayes Accuracy: ', nb.score(X_test, y_test))
    print('LogisticRegr Accuracy: ', logisticRegr.score(X_test, y_test))


























