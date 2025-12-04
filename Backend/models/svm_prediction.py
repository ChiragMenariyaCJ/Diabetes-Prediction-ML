import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC

classifier = svm.SVC(kernel='linear')


def predict(input_data, X_train, X_test, Y_train, Y_test):
    svm = SVC(kernel='linear')
    svm_predict = svm_accuracy = None
    svm.fit(X_train, Y_train)
    classifier.fit(X_train, Y_train)
    # accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    svm_predict = svm.predict(input_data)
    svm_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score on svm training data : ', svm_training_data_accuracy)
    X_test_prediction = classifier.predict(X_test)
    svm_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score on SVM the test data : ', svm_accuracy)
    final_prediction = (("{:.2f}%".format(round(svm_accuracy * 100, 2))), bool(svm_predict[0]))
    return final_prediction
