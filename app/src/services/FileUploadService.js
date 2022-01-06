import http from "../http-common";

const uploadSequence = (filename) => {
    var url = "http://localhost:5000/process_train/" + filename;
    return new EventSource(url);
    
};

const uploadPickle = (filename, sequence) => {
  var url = "http://localhost:5000/generate_map/" + filename + "/" + sequence;
  return new EventSource(url);
};

const getFiles = () => {
  return http.get("http://localhost:8080/files", {crossDomain: true});
};

export default {
  uploadSequence,
  uploadPickle,
  getFiles,
};