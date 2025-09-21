import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import NavDropdown from 'react-bootstrap/NavDropdown';
import axios from 'axios';
import ProfileAvatar from './profile/ProfileAvatar';
import UpdateUsernameModal from './profile/UpdateUsernameModal';
import './navbar.css';

const CustomNavbar = ({ isAuthenticated, setIsAuthenticated, username, setUsername, setmessage }) => {
  const [showModal, setShowModal] = useState(false);
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      const response = await axios.post('https://persuasive.research.cs.dal.ca/smsys/logout', {}, {
        withCredentials: true,
      });
      if (response.status === 200) {
        console.log("logged out");
        setIsAuthenticated(false);
        setmessage('not active session');
        setUsername(''); // Clear username
        navigate('/login');
      }
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const handleUpdateUsername = (newUsername) => {
    setUsername(newUsername);
  };

  return (
    <>
      <Navbar bg='dark' variant='dark' expand="lg" className="navbar-design">
        <Navbar.Brand as={Link} to="/" className="navbar-brand">SMSYS</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ml-auto">
            <Nav.Link as={Link} to="/">Home</Nav.Link>
            {isAuthenticated && <Nav.Link as={Link} to="/dashboard">Dashboard</Nav.Link>}
            {isAuthenticated && <Nav.Link as={Link} to="/detector">Detector</Nav.Link>}
            <Nav.Link as={Link} to="/about">About</Nav.Link>
            {isAuthenticated ? (
              <NavDropdown title={<ProfileAvatar username={username} />} id="basic-nav-dropdown">
                <NavDropdown.Item onClick={() => setShowModal(true)}>Profile</NavDropdown.Item>
                <NavDropdown.Item onClick={handleLogout}>Logout</NavDropdown.Item>
              </NavDropdown>
            ) : (
              <Nav.Link as={Link} to="/login">Login</Nav.Link>
            )}
          </Nav>
        </Navbar.Collapse>
      </Navbar>

      <UpdateUsernameModal
        show={showModal}
        handleClose={() => setShowModal(false)}
        currentUsername={username}
        updateUsernameInState={handleUpdateUsername}
      />
    </>
  );
};

export default CustomNavbar;
