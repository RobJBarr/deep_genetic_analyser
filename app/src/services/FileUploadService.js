import axios from "axios";
import httpCommon from "../http-common";
import http from "../http-common";

const upload = (file) => {
  let formData = new FormData();
  formData.append("file", file);

    var path = file.files[0].name;    
    var url = "http://localhost:5000/process_train/" + path + "/0";

    console.log(url);
    return new EventSource(url);
    
};

const getFiles = () => {
  return http.get("http://localhost:8080/files", {crossDomain: true});
};

export default {
  upload,
  getFiles,
};