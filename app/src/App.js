import React from "react";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";

import Main from "./components/Main"
function App() {
  return (
    <div className="container" style={{ width: "100%" }}>
      <div className="my-3">
        <div class="header">Deep Genetic Analyser</div>
      </div>

      <div className="App">
        <Main />
      </div>
    </div>
  );
}

export default App;