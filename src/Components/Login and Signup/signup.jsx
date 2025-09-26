import React from 'react';
import { useForm } from 'react-hook-form';
import { Link, useNavigate } from 'react-router-dom';
import "./login.css";

const Signup = ({ setIsAuthenticated, setUsername }) => {
  const { register, handleSubmit, formState: { errors }, watch } = useForm();
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = React.useState('');
  const password = watch('password', '');
  const confirmPassword = watch('confirmPassword', '');

  const onSubmit = async (data) => {
    if (data.password !== data.confirmPassword) {
      setErrorMessage('Passwords do not match');
      return;
    }
    try {
      const response = await fetch('https://persuasive.research.cs.dal.ca/smsys/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });

      const result = await response.json();
      if (response.ok) {
        // setIsAuthenticated(true);
        setUsername(data.username);
        navigate('/login');
      } else {
        setErrorMessage(result.message || 'Registration failed');
      }
    } catch (error) {
      setErrorMessage('An error occurred. Please try again.');
      console.error('Error:', error);
    }
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-form">
          <h2>Sign Up</h2>
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
              <input
                type="password"
                placeholder="Confirm Password"
                {...register('confirmPassword', { 
                  required: 'Confirm Password is required',
                  validate: (value) => value === password || 'Passwords do not match'
                })}
              />
              {errors.confirmPassword && <span className="error">{errors.confirmPassword.message}</span>}
              {confirmPassword && password && confirmPassword !== password && <span className="error">Passwords do not match</span>}
            </div>
            {errorMessage && <div className="error-message">{errorMessage}</div>}
            <div className="form-group">
              <button type="submit">Sign Up</button>
            </div>
          </form>
        </div>
        <div className="login-welcome">
          <h2>Welcome to Signup</h2>
          <p>Already have an account?</p>
          <Link to="/login">
            <button className="sign-up-button">Sign In</button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Signup;
