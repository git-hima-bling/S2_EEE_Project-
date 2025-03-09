import React from "react";
import ReactDOM from "react-dom/client";
import "./App.css"; // Import the CSS file for styling
import App from "./App"; // Import the main App component

// Get the root element from the DOM (public/index.html)
const root = ReactDOM.createRoot(document.getElementById("root"));

// Render the App component into the root element
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);