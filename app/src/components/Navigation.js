

import React from 'react';
 
import { NavLink } from 'react-router-dom';
 
const Navigation = () => {
    return (
       <div>
          <NavLink to="/">Home</NavLink>
          <NavLink to="/mutation_map">Mutation Map</NavLink>
       </div>
    );
}
 
export default Navigation;

