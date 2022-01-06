import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import FileUpload from '../pages/FileUpload';
import MutationMap from '../pages/MutationMap';

const Main = () => {
    console.log(useLocation().pathname)

  return (
    <Routes> 
      <Route exact path='/' element={<FileUpload/>}></Route>
      <Route exact path='/mutation_map' element={<MutationMap/>}></Route>
    </Routes>
  );
}

export default Main;