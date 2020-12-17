import pandas as pd
import numpy as np 
import csv

import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as sms
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns

def score_report(ytrain, y_hat):
    print("Accuracy Score: ", accuracy_score(ytrain, y_hat))
    print("Precision Score: ", precision_score(ytrain, y_hat)) 
    print("Recall Score: ", recall_score(ytrain, y_hat))  
    print("F1 Score: ", f1_score(ytrain, y_hat))
    pass

def getList(dict):
    return dict.keys()

def model_opt(models, x, y, xtest, ytest):
    """Purpose: iterates through list of clf and will evaluate model and display the confusion matrix"""
    for model in models:
        pipe = Pipeline(steps=[('model', model,)])
        fit = pipe.fit(x, y)
        ypred = model.predict(xtest)
        score_report(ytest, ypred)
        print(model," ", fit.score(xtest, ytest))
        plot_confusion_matrix(model, xtest, ytest, values_format='1')
        plt.show()
    pass

def single_model_opt(model, x, y, xtest, ytest):
    pipe = Pipeline(steps=[('model', model)])
    fit = pipe.fit(x, y)
    ypred = model.predict(xtest)
    score_report(ytest, ypred)
    print(model," ", fit.score(xtest, ytest))
    plot_confusion_matrix(model, xtest, ytest, values_format='1')
    plt.show()
    pass

def model_scoring(models, x_train_resampled, y_train_resampled, x_test, y_test):
    for model in models:
        train_score = model.score(x_train_resampled, y_train_resampled)
        test_score = model.score(x_test, y_test)
        avg_score = (train_score + test_score)/2
        print(model ,train_score, test_score, avg_score)

def cost_benefit_analysis(model, X_test, y_test):
    y_preds = model.predict(X_test)
    label_dict = {"TP":0, "FP": 0, "TN": 0, "FN": 0}
    for yt, yp in zip(y_test, y_preds):
        if yt==yp:
            if yt==1:
                label_dict["TP"] += 1
            else:
                label_dict["TN"] += 1
        else:
            if yp==1:
                label_dict["FP"] += 1
            else:
                label_dict["FN"] += 1
    cb_dict = {"TP": 50, "FP": -10, "TN": 0, "FN": -60}
    total = 0
    for key in label_dict.keys():
        total += cb_dict[key]*label_dict[key]
    return total / sum(label_dict.values())

from sklearn import metrics
import matplotlib.pyplot as plt

def evaluate_model(y_test,y_hat_test,X_test,clf=None,
                  scoring=metrics.recall_score,verbose=False,
                  scorer=False):
    """Quick/simple classification model evaluatiin"""

    print(metrics.classification_report(y_test,y_hat_test))
    
    metrics.plot_confusion_matrix(clf,X_test,y_test,
                                 cmap='Blues')
    plt.show()
#     if verbose:
#         print("MODEL PARAMETERS:")
#         print(pd.Series(rf.get_params()))
        
    if scorer:
        
        return scoring(y_test,y_hat_test)
    

def process_tweet(text,as_lemmas=False,as_tokens=True):
#     text=text.copy()
    for x in find_urls(text):
        text = text.replace(x,'')
        
    for x in find_retweets(text):
        text = text.replace(x,'')    
        
    for x in find_hashtags(text):
        text = text.replace(x,'')    

    if as_lemmas:
        from nltk.stem.wordnet import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        text = lemmatizer.lemmatize(text)
    
    if as_tokens:
        text = clean_text(text)
    
    if len(text)==0:
        text=''
            
    return text
