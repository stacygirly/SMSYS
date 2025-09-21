import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './about.css'; // Ensure you have appropriate styling

const About = ({ isAuthenticated }) => {
    const [showModal, setShowModal] = useState(false);
    const navigate = useNavigate();

    const handleDetector = () => {
        if (isAuthenticated) {
            navigate('/detector');
        } else {
            setShowModal(true);
        }
    };

    const handleLogin = () => {
        setShowModal(false);
        navigate('/login');
    };

    const handleClose = () => {
        setShowModal(false);
    };

    return (
        <>
            <div className="home">
                <div className="container">
                    <h3 style={{color:'white'}}>How to get started</h3>
                    <p style={{color:'white'}}>Here’s a quick overview of the process, from start to finish.</p>
                    <div className="instructions">
                        <div className="instruction-step">
                            <div className="step-number">1</div>
                            <h4>Click the "Start" button</h4>
                            <p>Click the "Start" button to begin recording your voice.</p>
                        </div>
                        <div className="instruction-step">
                            <div className="step-number">2</div>
                            <h4>Speak clearly</h4>
                            <p>Speak clearly into your microphone for a few seconds.</p>
                        </div>
                        <div className="instruction-step">
                            <div className="step-number">3</div>
                            <h4>Click the "Stop" button</h4>
                            <p>Click the "Stop" button to end the recording.</p>
                        </div>
                        <div className="instruction-step">
                            <div className="step-number">4</div>
                            <h4>Wait for analysis</h4>
                            <p>Wait for the system to analyze your voice and display whether you are stressed or not stressed.</p>
                        </div>
                        <div className="instruction-step">
                            <div className="step-number">5</div>
                            <h4>View results</h4>
                            <p>You can also view your previous results by clicking on the "Previous Results" section.</p>
                        </div>
                    </div>
                    <button className="cta-button" onClick={handleDetector}>Let's go</button>
                </div>
                <div className="about-system">
                    <h3 style={{color:'white'}}>About Our AI-Powered Stress Detector</h3>
                    <p style={{color:'white'}}>Our system utilizes advanced AI technology to accurately detect stress levels from voice recordings. Here’s how it works:</p>
                    <ul style={{color:'white'}}>
                        <li style={{color:'white'}}><strong>Data Collection:</strong> We collect voice samples and analyze various acoustic features such as pitch, tone, and pace.</li>
                        <li style={{color:'white'}}><strong>Feature Extraction:</strong> Key features that correlate with stress levels are extracted from the voice samples.</li>
                        <li style={{color:'white'}}><strong>Model Training:</strong> The extracted features are used to train a machine learning model using a large dataset of labeled voice recordings.</li>
                        <li style={{color:'white'}}><strong>Real-Time Analysis:</strong> When you record your voice, the system analyzes it in real-time using the trained model to determine your stress level.</li>
                        <li style={{color:'white'}}><strong>Accuracy:</strong> Our model is continuously updated and validated to ensure high accuracy in stress detection.</li>
                    </ul>
                    <p style={{color:'white'}}>We are committed to providing a reliable and easy-to-use tool to help you monitor and manage your stress levels effectively.</p>
                </div>
            </div>
            {showModal && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <h2 style={{color:'black'}}>Please Login First</h2>
                        <div className="modal-buttons">
                            <button onClick={handleLogin} className="modal-button login-button">Login</button>
                            <button onClick={handleClose} className="modal-button cancel-button">Cancel</button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default About;
