import React, { useState } from "react";
import { PredictDiabetesService } from "../../services/PredictDiabetes";

const DiabetesPrediction = () => {
  const [inputFormData, setInputFormData] = useState({
    pregnancies: "",
    glucose: "",
    bloodPressure: "",
    skinThickness: "",
    insulin: "",
    BMI: "",
    diabetesPedigreeFunction: "",
    age: "",
  });
  const [predictionData, setPredictionData] = useState(null);
  const [isDataLoading, setIsDataLoading] = useState(false);
  const [formError, setFormError] = useState(false);

  const inputDataHandler = (e) => {
    const { name, value } = e.target;
    setInputFormData({
      ...inputFormData,
      [name]: value,
    });
  };

  const resetFormData = () => {
    setInputFormData({
      pregnancies: "",
      glucose: "",
      bloodPressure: "",
      skinThickness: "",
      insulin: "",
      BMI: "",
      diabetesPedigreeFunction: "",
      age: "",
    });
  };

  const submitInputDataHandler = (e) => {
    e.preventDefault();
    const {
      pregnancies,
      glucose,
      bloodPressure,
      skinThickness,
      insulin,
      BMI,
      diabetesPedigreeFunction,
      age,
    } = inputFormData;

    // check if input field data is not empty
    if (
      pregnancies &&
      glucose &&
      bloodPressure &&
      skinThickness &&
      insulin &&
      BMI &&
      diabetesPedigreeFunction &&
      age
    ) {
      // set loading state before api call
      setIsDataLoading(true);
      // call api
      const payload = {
        input: Object.values(inputFormData).map((item) => Number(item)),
      };

      PredictDiabetesService.getPredictionData(payload)
        .then((res) => {
          console.log(res.data);
          setPredictionData(res.data);
          setIsDataLoading(false);
        })
        .catch((err) => {
          console.log(err);
          setIsDataLoading(false);
        });

      resetFormData();
      setFormError(false);
    } else {
      setFormError(true);
    }
  };

  return (
    <div className="row row-cols-lg-2">
      <div className="first">
        <div className="card p-5 d-flex flex-column">
          <h3>INPUT FORM</h3>
          <form onSubmit={submitInputDataHandler} className="">
            <div className="form-group">
              <label htmlFor="pregnancies">Pregnancies</label>
              <input
                type="number"
                className="form-control"
                id="pregnancies"
                name="pregnancies"
                value={inputFormData.pregnancies}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="glucose">Glucose</label>
              <input
                type="number"
                className="form-control"
                id="glucose"
                name="glucose"
                value={inputFormData.glucose}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="bloodPressure">Blood Pressure</label>
              <input
                type="number"
                className="form-control"
                id="bloodPressure"
                name="bloodPressure"
                value={inputFormData.bloodPressure}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="skinThickness">Skin Thickness</label>
              <input
                type="number"
                className="form-control"
                id="skinThickness"
                name="skinThickness"
                value={inputFormData.skinThickness}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="insulin">Insulin</label>
              <input
                type="number"
                className="form-control"
                id="insulin"
                name="insulin"
                value={inputFormData.insulin}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="BMI">BMI</label>
              <input
                type="number"
                className="form-control"
                id="BMI"
                name="BMI"
                value={inputFormData.BMI}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="diabetesPedigreeFunction">
                Diabetes Pedigree Function
              </label>
              <input
                type="number"
                className="form-control"
                id="diabetesPedigreeFunction"
                name="diabetesPedigreeFunction"
                value={inputFormData.diabetesPedigreeFunction}
                onChange={inputDataHandler}
              />
            </div>
            <div className="form-group">
              <label htmlFor="age">Age</label>
              <input
                type="number"
                className="form-control"
                id="age"
                name="age"
                value={inputFormData.age}
                onChange={inputDataHandler}
              />
            </div>
            {formError && (
              <p className="text-danger">
                Please fill out all the input fields
              </p>
            )}
            <button type="submit mb-4" className="btn btn-primary">
              Submit
            </button>
          </form>
        </div>
      </div>
      <div className={"sec" +( isDataLoading?" custom-container": "")}>
        {isDataLoading? (
          <div class="spinner-border custom-spinner" role="status">
            {/* <span class="sr-only">Loading...</span> */}
          </div>
        ) : (
          <>
            {predictionData && (
              <div className="card p-5 d-flex flex-column align-items-center">
                <h2>Prediction Result</h2>
                <p>
                  {predictionData.knn_prediction[1]
                    ? "Take care you are diabetic"
                    : "Congrats you are not diabetic"}
                </p>
                <button
                  className="btn btn-primary"
                  onClick={() => setPredictionData(null)}
                >
                  Reset
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default DiabetesPrediction;
