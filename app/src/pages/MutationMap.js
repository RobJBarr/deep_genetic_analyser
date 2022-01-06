import React, { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

import { Link } from "react-router-dom";

const MutationMap = () => {
    const [selectedFile, setSelectedFile] = useState("");
    const [sequence, setSequence] = useState("");
    
    const selectFile = (event) => {       
        for (const file of event.target.files) {
            setSelectedFile(file.name);
        }
        console.log(selectedFile)
    };

    const selectSequence = (event) => {
        setSequence(event.target.value);
        console.log(sequence)
    }

    const generateMap = () => {
        console.log(selectedFile)
        console.log(sequence)
        var eventSource = UploadService.uploadPickle(selectedFile, sequence);
        eventSource.onmessage = function (e) {
                console.log(e.data)
        }
    }

    return (
        <div class ="body">
            <div class="container">
                <h1>Upload Model</h1>
                <div class="form">
                    <label for="file">Model:</label>
                    <input type="file" name="file" onChange={selectFile}/>
                    <label for="sequence">Sequence:</label>
                        <input type="text" id="sequence" name="sequence" onChange={selectSequence}/>
                    <button class="training-button" onClick={generateMap} style={{display:"block"}}>Generate Mutation Map</button>
                </div>
            </div>
            <Link to="/">
                <button variant="outlined">
                    Train A Model
                </button>
            </Link>
        </div>
    );
};

export default MutationMap;