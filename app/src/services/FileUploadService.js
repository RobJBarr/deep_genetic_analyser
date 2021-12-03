import http from "../http-common";

const upload = (filename) => {
    var url = "http://localhost:5000/process_train/" + filename + "/0";

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