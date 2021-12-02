import React, { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

const UploadFiles = () => {

    const [selectedFiles, setSelectedFiles] = useState(undefined);
    const [currentFile, setCurrentFile] = useState(undefined);
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState("");

    const [fileInfos, setFileInfos] = useState([]);

    const selectFile = (event) => {
        setSelectedFiles(event.target.files);
        var progressbar = document.getElementById("progress-bar")
        progressbar.style.display = "block";
        var eventSource = UploadService.upload(event.target);
        eventSource.onmessage = function (e) {
            if (e.data === "100") {
                console.log("Finished Training")
                eventSource.close()
            } else {
                progressbar.value = e.data;
            }
        }
        progressbar.classList.add("color");
    };


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
        
    };

    // useEffect(() => {
    //     UploadService.getFiles().then((response) => {
    //         setFileInfos(response.data);
    //     });
    // }, []);
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
                {currentFile
                    }
            </div>

            

            
        </div>
    );
};

export default UploadFiles;