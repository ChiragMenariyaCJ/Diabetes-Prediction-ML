from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def predict(input_data, X_train, X_test, Y_train, Y_test):
    
    # accuracy score on the training data
    lr = LogisticRegression()
    lr_predict = lr_accuracy = None
    lr.fit(X_train, Y_train)
    X_train_prediction = lr.predict(X_train)
    lr_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score on lr  training data : ', lr_training_data_accuracy)
    X_test_prediction = lr.predict(X_test)
    lr_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score on LR the test data : ', lr_accuracy)    
    lr_predict = lr.predict(input_data)
    
    final_prediction = (("{:.2f}%".format(round(lr_accuracy * 100, 2))), bool(lr_predict[0]))
    return final_prediction
