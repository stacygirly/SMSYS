import React, { useState, useEffect } from 'react';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import axios from 'axios';
import './modal.css'; // Import the modal CSS

const UpdateUsernameModal = ({ show, handleClose, currentUsername, updateUsernameInState }) => {
  const [newUsername, setNewUsername] = useState(currentUsername);
  const [error, setError] = useState('');

  useEffect(() => {
    setNewUsername(currentUsername);
    setError('');
  }, [currentUsername, show]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (newUsername.trim() === '') {
      setError('Username cannot be empty');
      return;
    }
    
    try {
      const response = await axios.post('https://persuasive.research.cs.dal.ca/smsys/update_username', { new_username: newUsername }, { withCredentials: true });
      if (response.status === 200) {
        updateUsernameInState(newUsername);
        handleClose();
      }
    } catch (error) {
      setError("Failed to update username");
      console.error("Failed to update username:", error);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} centered>
      <Modal.Header closeButton>
        <Modal.Title>Profile Management</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h5>Current Username: {currentUsername}</h5>
        <Form onSubmit={handleSubmit}>
          <Form.Group controlId="formUsername">
            <Form.Label>New Username</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter new username"
              value={newUsername}
              onChange={(e) => {
                setNewUsername(e.target.value);
                setError(''); // Clear error when user starts typing
              }}
              isInvalid={!!error}
            />
            <Form.Control.Feedback type="invalid">
              {error}
            </Form.Control.Feedback>
          </Form.Group>
          <Button variant="primary" type="submit"  className="btn-primary-profile">
            Save Changes
          </Button>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default UpdateUsernameModal;
