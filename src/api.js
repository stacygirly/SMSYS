// src/api.js
import axios from "axios";

const api = axios.create({
  baseURL: "https://persuasive.research.cs.dal.ca/smsys",
  withCredentials: true,                       // ← send/receive cookies
  headers: { "Content-Type": "application/json" }
});

export default api;
