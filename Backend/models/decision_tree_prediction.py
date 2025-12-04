from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#function to predict the outcome based on input data and training/testing data
def predict(input_data, X_train, X_test, Y_train, Y_test):

    # training DecisionTreeClassifier model on training data and predicting on training data
    dtc = DecisionTreeClassifier()
    dtc_predict = dtc_accuracy = None
    dtc.fit(X_train, Y_train)
    X_train_prediction = dtc.predict(X_train)

    # calculating accuracy score on training data
    dtc_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score on DTC  training data : ', dtc_training_data_accuracy)

    # predicting on test data and calculating accuracy score on test data
    X_test_prediction = dtc.predict(X_test)
    dtc_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score on DTC the test data : ', dtc_accuracy)

    # calculating accuracy score on training data
    dtc_predict = dtc.predict(input_data)

    # returning final prediction as a tuple with accuracy score and predicted outcome
    final_prediction = (
        ("{:.2f}%".format(round(dtc_accuracy * 100, 2))), bool(dtc_predict[0]))
    return final_prediction
