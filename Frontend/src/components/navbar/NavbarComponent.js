import React from "react";

const NavbarComponent = () => {
  return (
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-3 w-100">
      <div class="container">
        <a class="navbar-brand" href="http://localhost:3000">
          TESTING 
        </a>
        <button
          class="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNavAltMarkup"
          aria-controls="navbarNavAltMarkup"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
          <div class="navbar-nav">
            <a class="nav-link active" aria-current="page" href="http://localhost:3000">
              Home
            </a>
            <a class="nav-link" href="http://localhost:3000">
              About
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavbarComponent;
