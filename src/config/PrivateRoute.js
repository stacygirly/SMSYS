import React from 'react';
import { Navigate } from 'react-router-dom';

const PrivateRoute = ({ isAuthenticated, children }) => {
  // if(isAuthenticated === true){
    // return <Navigate to="/"/>;
  // }

  // else {
  //   <Navigate to="/login"/>;
  // }
  return isAuthenticated ? children : <Navigate to="/login" />;
};

export default PrivateRoute;
