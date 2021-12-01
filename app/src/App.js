import React from "react";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";

import FileUpload from "./components/FileUpload";

function App() {
  return (
    <div className="container" style={{ width: "100%" }}>
      <div className="my-3">
        <div class="header">Deep Genetic Analyser</div>
      </div>

      <FileUpload />
    </div>
  );
}

export default App;