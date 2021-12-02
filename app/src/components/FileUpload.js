import React, { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

const UploadFiles = () => {

    const [selectedFiles, setSelectedFiles] = useState(undefined);
    const [currentFile, setCurrentFile] = useState(undefined);
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState("");

    const [fileInfos, setFileInfos] = useState([]);
    const uploadedFiles = ["test1.pdf", "test2.pdf"];

    const selectFile = (event) => {
        setSelectedFiles(event.target.files);
        upload();
        for (const file of event.target.files) {
            uploadedFiles.push(file.name);
        }
    };

    var checkedValue = "null";

    const uncheck = fileName => (event) => {
        var nullButtons = document.querySelectorAll('input[value="null"]');
        var trainingButton = document.querySelector(".training-button");
        if (checkedValue == fileName) {
            nullButtons.forEach(function (thisButton) {
                thisButton.checked = true;
            })
            trainingButton.disabled = true;
            trainingButton.classList.remove("animation");
            checkedValue = "null";
        } else {
            nullButtons.forEach(function (thisButton) {
                thisButton.checked = false;
            })
            trainingButton.disabled = false;
            trainingButton.classList.add("animation");
            checkedValue = fileName;
        }
    }

    // const deselectableRadios = (rootElement) => {
    //     if(!rootElement) rootElement = document;
    //     if(!window.radioChecked) window.radioChecked = {};
    //     window.radioClick = function(e) {
    //       const obj = e.target, name = obj.name || "unnamed";
    //       if(e.keyCode) return obj.checked = e.keyCode!=32;
    //       obj.checked = window.radioChecked[name] != obj;
    //       window.radioChecked[name] = obj.checked ? obj : null;
    //     }
    //     rootElement.querySelectorAll("input[type='radio']").forEach( radio => {
    //       radio.setAttribute("onclick", "radioClick(event)");
    //       radio.setAttribute("onkeyup", "radioClick(event)");
    //     });
    //   }

    const upload = () => {
        // let currentFile = selectedFiles[0];

        // setProgress(0);
        // setCurrentFile(currentFile);

        // UploadService.upload(currentFile, (event) => {
        //     setProgress(Math.round((100 * event.loaded) / event.total));
        // })
        //     .then((response) => {
        //         setMessage(response.data.message);
        //         return UploadService.getFiles();
        //     })
        //     .then((files) => {
        //         setFileInfos(files.data);
        //     })
        //     .catch(() => {
        //         setProgress(0);
        //         setMessage("Could not upload the file!");
        //         setCurrentFile(undefined);
        //     });

        // setSelectedFiles(undefined);
        document.getElementById("progress-bar").style.display = "block";
        var bar = document.getElementById("bar");
        bar.classList.add("color");
    };

    useEffect(() => {
        UploadService.getFiles().then((response) => {
            setFileInfos(response.data);
        });
    }, []);
    return (
        <div class="body">
            {currentFile && (
                <div className="progress">
                    <div
                        className="progress-bar progress-bar-info progress-bar-striped"
                        role="progressbar"
                        aria-valuenow={progress}
                        aria-valuemin="0"
                        aria-valuemax="100"
                        style={{ width: progress + "%" }}
                    >
                        {progress}%
                    </div>
                </div>
            )}

            <div id="progress-bar" class="progress" style={{display:'none'}}>
                <div id="bar"></div>
            </div>
            
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
                {currentFile
                    }
                <div class="uploaded-file-table">
                    {uploadedFiles.map((fileName) => (
                            <div class="uploaded-file-radio">
                                <input type="radio" name="files" value={fileName} onClick={uncheck(fileName)}/>
                                <label for={fileName}>{fileName}</label>
                            </div>
                        ))}
                    <input type="radio" name="files" value="null" style={{display:"none"}}></input>
                </div>
            </div>
            <button class="training-button" disabled>Begin Training</button>
        </div>
    );
};

export default UploadFiles;