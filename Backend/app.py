from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models import decision_tree_prediction
from models import knn_prediction
from models import logistic_regression_prediction
from models import random_forest_prediction
from models import svm_prediction

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
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

    df["Outcome"].value_counts()
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']

    # scaler = StandardScaler()
    # scaler.fit(X)
    # standardized_data = scaler.transform(X)

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = df['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)
    # Get the percentage of diabetics


    data = request.json
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(data["input"])

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    input_data = std_data
    print(std_data)

    # data["input"] = (6,148,72,35,0,33.6,0.627,50)
    # input_data = np.array(data['input'])
    # input_data = input_data.reshape(1, -1)
    # input_data = scaler.transform(input_data)
    # print("input_data ----------->", input_data)
#     input_data_as_numpy_array = np.asarray(input_data)

# # reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# # standardize the input data
# std_data = scaler.transform(input_data_reshaped)

    final_svm_prediction = svm_prediction.predict(input_data, X_train, X_test, Y_train, Y_test)
    final_lr_prediction = logistic_regression_prediction.predict(input_data, X_train, X_test, Y_train, Y_test)
    final_knn_prediction = knn_prediction.predict(input_data, X_train, X_test, Y_train, Y_test)
    final_dtc_prediction = decision_tree_prediction.predict(input_data, X_train, X_test, Y_train, Y_test)
    final_rfc_prediction = random_forest_prediction.predict(input_data, X_train, X_test, Y_train, Y_test)

    # final_rf_prediction = random_forest.predict(input_data, X_train, X_test, Y_train, Y_test)
    pred = {"svm_prediction":final_svm_prediction, 
            "lr_prediction": final_lr_prediction,
            "knn_prediction": final_knn_prediction,
            "dtc_prediction": final_dtc_prediction,
            "rfc_prediction": final_rfc_prediction,
            }
    print("Final Prediction Response:", pred)
    return {"knn_prediction": final_knn_prediction}

if __name__ == '__main__':
    app.run(debug=True)