from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def predict(input_data, X_train, X_test, Y_train, Y_test):
    
    # accuracy score on the training data
    knn = KNeighborsClassifier()
    knn_predict = knn_accuracy = None
    knn.fit(X_train, Y_train)
    X_train_prediction = knn.predict(X_train)
    knn_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score on KNN  training data : ', knn_training_data_accuracy)
    X_test_prediction = knn.predict(X_test)      
    knn_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score on KNN the test data : ', knn_accuracy)
    knn_predict  = knn.predict(input_data)
    final_prediction = (("{:.2f}%".format(round(knn_accuracy * 100, 2))), bool(knn_predict[0]))
    return final_prediction
