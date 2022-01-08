import http from "../http-common";
import axios from "axios";


const getModel = (filename, data) => {
    var isWaiting = true
    uploadSequence(data)
        .then(response => {console.log(response);
          isWaiting = false
        })
    if (isWaiting == true){
        console.log("waiting")
    }
    
    var url = "http://localhost:5000/process_train/" + filename;
    
    return url;
    
    
    
};



async function uploadSequence(data){
  let file = data
  const formData = new FormData();
  formData.append("file", file);
  const headers = {
    "Access-Control-Allow-Origin": "*"
  }
  
  return axios
    .post("http://localhost:5000/process_sequence", formData, {
      headers: headers
    })
}

async function uploadPickle(data){
    let file = data
    const formData = new FormData();
  
    formData.append("file", file);
  
    axios
      .post("http://localhost:5000/process_pickle", formData)
      .then(res => console.log(res))
      .catch(err => console.warn(err));
      
}

async function getMutationMap(filename, sequence, data){
  await uploadPickle(data).then(resp => {console.log(resp)})
  var url = "http://localhost:5000/generate_map/" + filename + "/" + sequence;

  return await axios.get(url).then(response => {
    if (response) {
      return response.data
    }
    return Promise.reject('An unknown error occurred');
  });
  
};



const getFiles = () => {
  return http.get("http://localhost:8080/files", {crossDomain: true});
};

export default {
  getModel,
  getMutationMap,
  uploadSequence,
  uploadPickle,
  getFiles,
};