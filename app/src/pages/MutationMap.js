import React, { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

import { Link } from "react-router-dom";

const MutationMap = () => {
    const [selectedFile, setSelectedFile] = useState("");
    const [sequence, setSequence] = useState("");
    const [selectedContents, setSelectedContents] = useState(undefined);
    const selectFile = (event) => {       
        for (const file of event.target.files) {
            setSelectedFile(file.name);
            setSelectedContents(file)
        }
        console.log(selectedFile)
    };

    const selectSequence = (event) => {
        setSequence(event.target.value);
        console.log(sequence)
    }

    const generateMap = () => {
        UploadService.getMutationMap(selectedFile, sequence, selectedContents).then(image => {
            var a = document.createElement("a"); //Create <a>
            a.download = "figure.png"; //File name Here
            a.value = "Click here"
            a.type = "image/png"
            a.href = 'data:image/png;base64,' + image;
            console.log(a.href)
            a.click();})
         //Downloaded file
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
                    <img id="my-img"/>
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