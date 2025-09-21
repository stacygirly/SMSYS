import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { Link, useNavigate } from 'react-router-dom';
import './login.css';

const Login = ({ theme, setTheme, setIsAuthenticated, isAuthenticated, onLogin }) => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const onSubmit = async (data) => {
    try {
      const success = await onLogin(data.username, data.password);
      if (success) {
        navigate('/'); // Navigate to home page on successful login
      } else {
        setErrorMessage('Invalid credentials. Please try again.');
      }
    } catch (error) {
      console.error('Login failed:', error);
      setErrorMessage('Invalid credentials. Please try again.');
    }
  };

  return (
    <div className={`login-page ${theme}`}>
      <div className="login-container">
        <div className="login-form">
          <h2>Sign In</h2>
          <form onSubmit={handleSubmit(onSubmit)}>
            <div className="form-group">
              <input
                type="text"
                placeholder="Username"
                {...register('username', { required: 'Username is required' })}
              />
              {errors.username && <span className="error">{errors.username.message}</span>}
            </div>
            <div className="form-group">
              <input
                type="password"
                placeholder="Password"
                {...register('password', {
                  required: 'Password is required',
                  minLength: { value: 6, message: 'Password must be at least 6 characters long' }
                })}
              />
              {errors.password && <span className="error">{errors.password.message}</span>}
            </div>
            <div className="form-group">
              <button type="submit">Sign In</button>
            </div>
            {errorMessage && <div className="error-message">{errorMessage}</div>}
            <div className="form-group">
              <Link to="/forgotpassword" className="forgot-password">Forgot Password?</Link>
            </div>
          </form>
        </div>
        <div className="login-welcome">
          <h2>Welcome to Login</h2>
          <p style={{color: 'white'}}>Don't have an account?</p>
          <Link to="/signup">
            <button className="sign-up-button">Sign Up</button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
