from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC

classifier = svm.SVC(kernel='linear')

app = Flask(__name__)

# Load the dataset
df3 = pd.read_csv("../Diabetes-prediction-using-ML/helper/diabetes.csv")
df1 = pd.read_csv("../Diabetes-prediction-using-ML/helper/diabetes1.csv")
df2 = pd.read_csv("../Diabetes-prediction-using-ML/helper/diabetes2.csv")
df4 = pd.read_csv("../Diabetes-prediction-using-ML/helper/diabetes3.csv")
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Preprocess the data
diabetes_df_copy = df.copy(deep=True)
diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df_copy[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
diabetes_df_copy['Glucose'].fillna(
    diabetes_df_copy['Glucose'].mean(), inplace=True)
diabetes_df_copy['BloodPressure'].fillna(
    diabetes_df_copy['BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna(
    diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin'].fillna(
    diabetes_df_copy['Insulin'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)
X = diabetes_df_copy.drop(columns='Outcome', axis=1)
Y = diabetes_df_copy['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)
# Get the percentage of diabetics
diabetic_percentage = df['Outcome'].value_counts(normalize=True)[1] * 100
print(
    f"The percentage of diabetics in the dataset is {diabetic_percentage:.2f}%")

# Check if the dataset has any diabetics
has_diabetes = df['Outcome'].unique().tolist() == [0, 1]
if has_diabetes:
    print("The dataset has diabetics")
else:
    print("The dataset does not have any diabetics")

# Train the classifiers
svm = SVC(kernel='linear')
svm.fit(X_train, Y_train)
classifier.fit(X_train, Y_train)
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
svm_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on svm training data : ', svm_training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
svm_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on SVM the test data : ', svm_test_data_accuracy)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
X_train_prediction = lr.predict(X_train)
lr_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on lr  training data : ', lr_training_data_accuracy)
X_test_prediction = lr.predict(X_test)
lr_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on LR the test data : ', lr_test_data_accuracy)


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
X_train_prediction = knn.predict(X_train)
knn_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on KNN  training data : ', knn_training_data_accuracy)
X_test_prediction = knn.predict(X_test)
knn_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on KNN the test data : ', knn_test_data_accuracy)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
X_train_prediction = dtc.predict(X_train)
dtc_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on DTC  training data : ', dtc_training_data_accuracy)
X_test_prediction = dtc.predict(X_test)
dtc_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on DTC the test data : ', dtc_test_data_accuracy)

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
X_train_prediction = rfc.predict(X_train)
rfc_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on RFC  training data : ', rfc_training_data_accuracy)
X_test_prediction = rfc.predict(X_test)
rfc_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on RFC the test data : ', rfc_test_data_accuracy)


@app.route('/', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input'])
    input_data = input_data.reshape(1, -1)
    input_data = scaler.transform(input_data)

    svm_prediction = svm.predict(input_data)
    lr_prediction = lr.predict(input_data)
    knn_prediction = knn.predict(input_data)
    dtc_prediction = dtc.predict(input_data)
    rfc_prediction = rfc.predict(input_data)
    predictions = {'Diabetic Percentage in Dataset': ("{:.2f}%".format(round(diabetic_percentage * 1, 2))),
                   'SVM': (("{:.2f}%".format(round(svm_test_data_accuracy * 100, 2))), bool(svm_prediction[0])),
                   'Logistic Regression': (("{:.2f}%".format(round(lr_test_data_accuracy * 100, 2))), bool(lr_prediction[0])),
                   'KNN': (("{:.2f}%".format(round(knn_test_data_accuracy * 100, 2))), bool(knn_prediction[0])),
                   'Decision Tree': (("{:.2f}%".format(round(dtc_test_data_accuracy * 100, 2))), bool(dtc_prediction[0])),
                   'Random Forest': (("{:.2f}%".format(round(rfc_test_data_accuracy * 100, 2))), bool(rfc_prediction[0])),
                   #    "LR Testing Accuracy": round(lr_test_data_accuracy * 100, 2)
                   }

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
