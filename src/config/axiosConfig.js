// src/axiosConfig.js
import axios from 'axios';

axios.defaults.withCredentials = true;
axios.defaults.baseURL = 'https://stress-detector.onrender.com/';

export default axios;
