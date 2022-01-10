import React, { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

import { Link } from "react-router-dom";



const FileUpload = () => {

    const [selectedFiles, setSelectedFiles] = useState([]);
    const [selectedFileNames, setSelectedFileNames] = useState([]);

    useEffect(() => {
        refreshRadios()
        
    }, [selectedFileNames]);
    const [currentFile, setCurrentFile] = useState(undefined);
    const [numFiles, setNumFiles] = useState(0);
    const [message, setMessage] = useState("");
    const [checkedValue, setCheckedValue] = useState(undefined);
    const [checkedContent, setCheckedContents] = useState(undefined)
    const [fileInfos, setFileInfos] = useState([]);
    const uploadedFiles = []

    const selectFile = (event) => {
        setSelectedFiles(oldArr => [...oldArr, event.target.files[0]])
        setSelectedFileNames(oldArr => [...oldArr, event.target.files[0].name])
        for (const file of event.target.files) {
                uploadedFiles.push(file);
        }
    };


    const sleep = (milliseconds) => {
        return new Promise(resolve => setTimeout(resolve, milliseconds))
      }

    const beginTraining = () => (event) => {
        if (checkedValue != null){
            var trainingButton = document.querySelector(".training-button");
            trainingButton.textContent = "Training...";
            trainingButton.disabled = true;
            var progressbar = document.getElementById("progress-bar")
            progressbar.style.display = "block";
            UploadService.getModel(checkedValue, checkedContent).then((url) => {
            var eventSource = new EventSource(url)
                eventSource.onerror = function() {
                    eventSource.close()
                };
            var penultimate = false
            eventSource.onmessage = function(e){
                if (penultimate) {
                    sleep(1000).then(() => {
                    var array = JSON.parse(e.data)
                    let bytes_array = new Uint8Array(array); //<--- add this
                    let mime_type = '{{application/octet-stream}}';
                    var blob = new Blob([bytes_array], { type: mime_type })
                    trainingButton.disabled = false;
                    var dlink = document.createElement('a');
                    dlink.download = 'trained_model.pickle';
                    dlink.href = window.URL.createObjectURL(blob);
                    dlink.onclick = function (e) {
                        console.log("Downloading")
                        // revokeObjectURL needs a delay to work properly.
                        var that = this;
                        setTimeout(function () {
                        window.URL.revokeObjectURL(that.href);
                        }, 1500);
                    };
                    document.body.appendChild(dlink);
                    dlink.click();
                    dlink.remove();
                    eventSource.close()})
                    
                    progressbar.value = '0';
                    progressbar.innerText = "";
                    trainingButton.textContent = "Train";
                    progressbar.style.display = "none";
                } else {
                    if (e.data.includes('100')){
                        penultimate = true
                    }
                    progressbar.value = e.data;
                    progressbar.innerText = e.data + "%";
                }
            }
            
            progressbar.classList.add("color")});}
        
    }
    

    const uncheck = fileName => (event) => {
        
        var nullButtons = document.querySelectorAll('input[value="null"]');
        var trainingButton = document.querySelector(".training-button");
            nullButtons.forEach(function (thisButton) {
                thisButton.checked = false;
            })
        trainingButton.classList.add("animation");
        setCheckedValue(event.target.value);
        for (var i = 0; i < selectedFiles.length; i++){
            if (selectedFiles[i].name === event.target.value){
                setCheckedContents(selectedFiles[i])
            }
        }
        
    }
    

    const refreshRadios = () => {
        var wrapper = document.getElementById('file-table');
        var elementsToInsert = [];
        for(var i = 0; i < selectedFileNames.length; i++) {
            var radio = document.createElement('input');
            var label = document.createElement('label');
            radio.type = 'radio';
            radio.name = 'files';
            radio.value = selectedFileNames[i];
            radio.onchange = uncheck(radio.value)
            label.setAttribute("for", selectedFileNames[i]);
            label.innerHTML = selectedFileNames[i];
            elementsToInsert.push({ radio: radio , label: label });
          }
          wrapper.innerHTML = "";
          for(var i = 0; i < elementsToInsert.length; i++) {
          
            // Array.prototype.splice removes items from the Array and return the an array containing the removed items (See https://www.w3schools.com/jsref/jsref_splice.asp)
            var toInsert = elementsToInsert[i];
            wrapper.appendChild(toInsert.radio);
            wrapper.appendChild(toInsert.label);
            wrapper.appendChild(document.createElement("br"));
          }
    }

    

    useEffect(() => {
        var trainingButton = document.querySelector(".training-button");
        if (checkedValue != null) {
            trainingButton.style = {"display":"block"};
        }
        
    }, [checkedValue]);



    return (
        <div class="body">        
            <progress id="progress-bar"value="0" max="100" style={{display:'none', width:"100%"}}/>
            <div class="upload">
            
                <div class="dropZoneContainer">  
                    <input type="file" id="drop_zone" class="FileUpload" onChange={selectFile} />
                    <div class="dropZoneOverlay">Upload For Secure Analysis</div>
                    <div class="dropZoneIcon"/>
                </div>
            
            <div className="alert alert-light" role="alert" style={{opacity:'0%'}}>
                {message}
            </div>
            </div>
            <div class="uploaded-files">
                <div className="card">
                    <div className="card-header">Your uploads</div>
                    <ul className="list-group list-group-flush">
                    </ul>
                </div>
                <div class="uploaded-file-table" id="file-table">
                </div>
                <button class="training-button" onClick={beginTraining()} style={{display:"none"}}>Begin Training</button>
            </div>
            <Link to="/mutation_map">
                <button variant="outlined">
                    Get A Mutation Map
                </button>
            </Link>
        </div>
    );
};

export default FileUpload;