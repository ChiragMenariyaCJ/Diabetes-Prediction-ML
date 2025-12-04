from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def predict(input_data, X_train, X_test, Y_train, Y_test):
    
    # accuracy score on the training data
    rfc = RandomForestClassifier()
    rfc_predict = rfc_accuracy = None
    rfc.fit(X_train, Y_train)
    X_train_prediction = rfc.predict(X_train)
    rfc_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score on RFC  training data : ', rfc_training_data_accuracy)
    X_test_prediction = rfc.predict(X_test)
    rfc_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score on RFC the test data : ', rfc_accuracy)
    rfc_predict = rfc.predict(input_data)
    
    final_prediction = (("{:.2f}%".format(round(rfc_accuracy * 100, 2))), bool(rfc_predict[0]))
    return final_prediction
