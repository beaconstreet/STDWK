import React from "react";
import ReactDOM from "react-dom/client";
import "../styles/globals.css";

const Popup: React.FC = () => {
  return (
    <div className="w-80 p-4">
      <h1 className="text-2xl font-bold mb-4">ZenFill</h1>
      <button className="btn btn-primary w-full">Fill This Page</button>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Popup />
  </React.StrictMode>
);
