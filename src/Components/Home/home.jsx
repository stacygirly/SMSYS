import React from 'react';
import {Link} from 'react-router-dom';
import './home.css';

const Home = () => {
    return (
        <>
            <div className='home-container'>

                <h1> Stress Monitoring System</h1>
                <p>detect stress with our AI powered model</p>
                <Link to ='/about'>
                <button className='learn-more-button'>LEARN MORE</button>
                </Link>
            </div>
        </>
    );
};

export default Home;
