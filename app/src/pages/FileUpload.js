import React, { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

import { Link } from "react-router-dom";



const FileUpload = () => {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [currentFile, setCurrentFile] = useState(undefined);
    const [message, setMessage] = useState("");
    const [checkedValue, setCheckedValue] = useState(undefined);
    const [checkedContent, setCheckedContents] = useState(undefined)
    const [fileInfos, setFileInfos] = useState([]);
    const uploadedFiles = [];
    const [optimalParams, setOptimalParams] = useState(undefined);
    const selectFile = (event) => {
        setSelectedFiles(oldArray => [...oldArray, event.target.files[0]])
        for (const file of event.target.files) {
                uploadedFiles.push(file);
        }
        refreshRadios();
    };

    const beginTraining = () => (event) => {
        console.log(selectedFiles)
        console.log(checkedContent)
        if (checkedValue != null){
            var trainingButton = document.querySelector(".training-button");
            trainingButton.textContent = "Training...";
            trainingButton.disabled = true;
            var progressbar = document.getElementById("progress-bar")
            progressbar.style.display = "block";
            var eventSource = UploadService.getModel(checkedValue, checkedContent);
            eventSource.onmessage = function (e) {
                if (e.data.includes('100')) {
                    console.log("Finished Training")
                    eventSource.close()
                    let bytes_array = new Uint8Array(e.data); //<--- add this
                    let mime_type = '{{application/octet-stream}}';
                    var blob = new Blob([bytes_array], { type: mime_type })
                    document.body.innerHTML +=
                        `<a id="download" download="model.pickle" href="./static/for_client/model.pickle"> Click me</a>`
                    trainingButton.disabled = false;
                    var dlink = document.createElement('a');
                    dlink.download = 'pickle.pickle';
                    dlink.href = window.URL.createObjectURL(blob);
                    dlink.onclick = function (e) {
                        // revokeObjectURL needs a delay to work properly.
                        var that = this;
                        setTimeout(function () {
                        window.URL.revokeObjectURL(that.href);
                        }, 1500);
                    };
                    document.body.appendChild(dlink);
                    dlink.click();
                    dlink.remove();
                } else {
                    progressbar.value = e.data;
                }
            }
            progressbar.classList.add("color");
        }
    }
    

    const uncheck = fileName => (event) => {
        
        var nullButtons = document.querySelectorAll('input[value="null"]');
        var trainingButton = document.querySelector(".training-button");
        console.log(event.target.value);
        if (checkedValue === event.target.value) {
            nullButtons.forEach(function (thisButton) {
                thisButton.checked = true;
            })
            trainingButton.style = {"display":"none"};
            trainingButton.classList.remove("animation");
        } else {
            nullButtons.forEach(function (thisButton) {
                thisButton.checked = false;
            })
            trainingButton.classList.add("animation");
            setCheckedValue(event.target.value);
            console.log(event.target.value)
            console.log("asdfasdf")
            for (var i = 0; i < selectedFiles.length; i++){
                console.log("adf")
                console.log(selectedFiles[i].name)
                if (selectedFiles[i].name === event.target.value){
                    
                    setCheckedContents(selectedFiles[i])
                }
            }
        }
    }
    

    const refreshRadios = () => {
        var wrapper = document.getElementById('file-table');
        var elementsToInsert = [];
        for(var i = 0; i < uploadedFiles.length; i++) {
            var radio = document.createElement('input');
            var label = document.createElement('label');
            radio.type = 'radio';
            radio.name = 'files';
            radio.value = uploadedFiles[i].name;
            radio.onchange = uncheck(radio.value)
            label.setAttribute("for", uploadedFiles[i].name);
            label.innerHTML = uploadedFiles[i].name;
          
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
            {currentFile}
        
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
                        {fileInfos &&
                            fileInfos.map((file, index) => (
                                <li className="list-group-item" key={index}>
                                    <a href={file.url}>{file.name}</a>
                                </li>
                            ))}
                    </ul>
                </div>
                {currentFile}
                <div class="uploaded-file-table" id="file-table">
                    {uploadedFiles.map((fileName) => (
                            <div class="uploaded-file-radio">
                                <input type="radio" name="files" value={fileName} onClick={uncheck(fileName)}/>
                                <label for={fileName}>{fileName}</label>
                            </div>
                        ))}
                    <input type="radio" name="files" value="null" style={{display:"none"}}></input>
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