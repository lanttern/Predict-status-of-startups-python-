# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:30:02 2015

@author: zhihuixie
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
     f1_score, make_scorer
from sklearn.grid_search import GridSearchCV

class Modeling():
    """
    This class is applied to spearate training and test dataset (train_test_split)
    , transform categorical features using DictVectorizer (feature_transform),
    prediction (predict) and tune parameters(tune_parameter)
    """
    def __init__(self, df, est):
        """
        Inititate two parameters: df - a pandas data fram including features 
        and labels, est- a machine learning estimator,
        and two functions: split dataset, transform features
        """
        self.df = df
        self.est = est
        self.train_test_split()
        self.feature_transform()
    def train_test_split(self):
        """
        This function is used to split training and testing dataset
        """
        self.labels = self.df.status
        # drop label column
        self.df1 = self.df.drop("status", axis = 1, inplace = False)
        # transform labels
        self.lbl = LabelEncoder()
        self.labels = self.lbl.fit_transform(self.labels)
        self.labels = self.labels.astype(float)
        # split train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = \
                                   train_test_split(self.df1, self.labels, \
                                   test_size = 0.35, random_state = 42)
        return self.x_train, self.x_test, self.y_train, self.y_test
    def feature_transform(self):
        """
        The features in train and test dataset are transformed by DictVectorizer
        """
        # convert dataframe to dictornary
        self.x_train = self.x_train.T.to_dict().values()
        self.x_test = self.x_test.T.to_dict().values()
        # transform
        dv = DictVectorizer()
        self.x_train = dv.fit_transform(self.x_train)
        self.x_test = dv.transform(self.x_test)
        self.x_train = self.x_train.astype(float)
        self.x_test = self.x_test.astype(float)
    def predict(self):
        """
        predict classification and probability
        """
        # fit model
        (clf, name) = self.est
        clf.fit(self.x_train, self.y_train)
        # predict classification
        preds = clf.predict(self.x_test)
        # predict probability
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(self.x_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(self.x_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        return preds, prob_pos
    def metrics(self, preds):
        """
        build the following metrics for modeling:
        accuracy_score: Accuracy classification score
        precision_score: the ability of the classifier not to label as positive 
        a sample that is negative
        recall_score:the ability of the classifier to find all the positive samples
        f1_score: a weighted average of the precision and recall
        """
        # get predictions and probability pf prediction
        (_, name) = self.est
        # print metrics score
        accuracy = accuracy_score(self.y_test, preds, normalize=True)
        precision = precision_score(self.y_test, preds, average = "weighted")
        recall = recall_score(self.y_test, preds, average = "weighted")
        f1 = f1_score(self.y_test, preds, average = "weighted")
        print "%s:" % name
        print "Accuracy score: %f" % accuracy
        print "Precision score: %f" % precision
        print "Recall score: %f" % recall
        print "F1 score: %f\n" % f1
        return {name:(accuracy, precision, recall, f1)}
    def tune_paramter(self, params):
        """
        this function is used to tune parameters for a estimator. In this modeling
        the labels are skewd/biased. So, the f1_score is used as metrics for tuning
        parameters
        """
        # get data
        clf,name = self.est
        # set up search
        best_clf = GridSearchCV(estimator = clf, param_grid = params, \
                   scoring = make_scorer(f1_score, average = "weighted"))
        # modeling with best parameters
        best_clf.fit(self.x_train, self.y_train)
        preds = best_clf.predict(self.x_test)
        return preds
        
    def inverse_labels(self, preds):
        """
        convert numeric predictions to be categories
        """
        return self.lbl.inverse_transform(preds.astype(int))

if __name__ == "__main__":
    # load data
    df = pd.read_csv("cleaned startups data_2013.csv")
    # try two algorithm
    ests = [(LinearSVC(random_state = 42), "Linear SVC"), \
            (LogisticRegression(multi_class="multinomial", solver = "lbfgs"), \
            "Logistic Regression")]
    # compare linearSVC and LogisticRegression    
    scores = {}
    for est in ests:
        model = Modeling(df, est)
        preds, _ = model.predict()
        score = model.metrics(preds)
        scores.update(score)
    print "Metrics for ests: \n", scores, "\n"
    print "The est with max accuracy score is: ",\
          max(scores, key = lambda x: scores[x][0]), "\n"
    print "The est with max precision score is: ",\
          max(scores, key = lambda x: scores[x][1]), "\n"
    print "The est with max recall score is: ",\
          max(scores, key = lambda x: scores[x][2]), "\n"
    print "The est with max f1 score is: ",\
          max(scores, key = lambda x: scores[x][3]), "\n"
    # tune parameters for LogisticRegression because it shows better metrics scores
    params = {"tol": [0.001, 0.0001, 0.00001, 0.000001], \
              "C": [1.0, 10, 100, 1000], "class_weight": [None, "balanced"], \
              "random_state":[None, 42], "max_iter": [10, 100, 1000], \
              "verbose":[0, 1, 10]}
    model = Modeling(df, ests[1])
    preds = model.tune_paramter(params)
    # print metrics scores for best parameters
    model.metrics(preds)
          
