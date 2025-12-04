import React from "react";
import DiabetesPrediction from "./components/diabetes/DiabetesPrediction";
// import About from "./components/about/About&Contact";
import NavbarComponent from "./components/navbar/NavbarComponent";
import "../src/assets/css/main.css";

function App() {
  return (
    <div className= "background-image">
    <div className="content">
        <NavbarComponent />
      <div className="container">
        <div className="mt-5 text-black">
            <DiabetesPrediction />
          </div>
        <div className="row justify-content-center align-items-center w-100">
            <div className="col-md-8">
              <h6 className="mt-5 text-center text-black">
                &copy; 2023. All Rights Reserved By SPSU.
              </h6>
              {/* <About&Contact /> */}
            </div>
          </div>
      </div>
    </div>
    </div>
  );
}

export default App;
