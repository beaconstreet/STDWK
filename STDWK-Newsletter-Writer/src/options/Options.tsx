import React from "react";
import ReactDOM from "react-dom/client";
import "../styles/globals.css";

const Options: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">ZenFill Options</h1>
      <div className="tabs tabs-boxed">
        <a className="tab tab-active">Your Information</a>
        <a className="tab">Form History</a>
        <a className="tab">About</a>
      </div>
      <div className="mt-4">{/* Form sections will be added here */}</div>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Options />
  </React.StrictMode>
);
