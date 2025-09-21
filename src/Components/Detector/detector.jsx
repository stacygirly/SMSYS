// import React, { useState, useRef, useEffect } from 'react';
// import axios from 'axios';
// import './detector.css'; // Importing the CSS file

// const Detector = ({ username }) => {
//     const [isRecording, setIsRecording] = useState(false);
//     const [emotion, setEmotion] = useState('');
//     const [details, setDetails] = useState('');
//     const [userEmotions, setUserEmotions] = useState([]);
//     const [filter, setFilter] = useState('All');
//     const [interval, setIntervalTime] = useState(1); // Default 1 minute in minutes
//     const [feedback, setFeedback] = useState(''); // State to track feedback
//     const mediaRecorderRef = useRef(null);
//     const intervalRef = useRef(null);
//     const isRecordingActive = useRef(false); // Use useRef to manage recording state
//     const audioContextRef = useRef(null); // Use ref to manage AudioContext
//     const streamRef = useRef(null); // Reference to the stream

//     const fetchUserEmotions = async () => {
//         try {
//             const response = await axios.get('http://localhost:5000/user_emotions', { withCredentials: true });
//             console.log("Fetched user emotions:", response.data); // Log backend response
//             setUserEmotions(response.data);
//         } catch (error) {
//             console.error("Error fetching user emotions:", error);
//         }
//     };

//     useEffect(() => {
//         fetchUserEmotions();
//         // Check the recording status when the component is mounted
//         chrome.runtime.sendMessage({ action: 'getStatus' }, (response) => {
//             setIsRecording(response.isRecording);
//         });
//         // Clear the badge when the popup is opened
//         chrome.runtime.sendMessage({ action: 'clearBadge' }, (response) => {
//             console.log(response.status); // Log the response for debugging
//         });
//     }, []);

//     const startRecording = async () => {
//         console.log(username);
//         setFeedback('Recording...'); // Set feedback message to "Recording..."

//         try {
//             // Request access to the user's microphone
//             const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//             streamRef.current = stream; // Store stream reference

//             // Initialize the AudioContext for pitch visualization
//             if (!audioContextRef.current) {
//                 audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
//             }

//             const analyser = audioContextRef.current.createAnalyser();
//             const source = audioContextRef.current.createMediaStreamSource(stream);
//             source.connect(analyser);

//             const canvas = document.getElementById('pitchCanvas');
//             const canvasContext = canvas.getContext('2d');

//             analyser.fftSize = 2048;
//             const bufferLength = analyser.fftSize;
//             const dataArray = new Uint8Array(bufferLength);

//             // Set recording state to active
//             isRecordingActive.current = true;

//             const drawPitch = () => {
//                 if (!isRecordingActive.current) return; // Stop drawing if recording is stopped
//                 requestAnimationFrame(drawPitch);

//                 analyser.getByteTimeDomainData(dataArray);

//                 // Set canvas background color
//                 canvasContext.fillStyle = '#3A4A6B';
//                 canvasContext.fillRect(0, 0, canvas.width, canvas.height);

//                 // Set waveform color
//                 canvasContext.lineWidth = 2;
//                 canvasContext.strokeStyle = '#B0C4DE';

//                 canvasContext.beginPath();

//                 const sliceWidth = canvas.width * 1.0 / bufferLength;
//                 let x = 0;

//                 for (let i = 0; i < bufferLength; i++) {
//                     const v = dataArray[i] / 128.0;
//                     const y = v * canvas.height / 2;

//                     if (i === 0) {
//                         canvasContext.moveTo(x, y);
//                     } else {
//                         canvasContext.lineTo(x, y);
//                     }

//                     x += sliceWidth;
//                 }

//                 canvasContext.lineTo(canvas.width, canvas.height / 2);
//                 canvasContext.stroke();
//             };

//             drawPitch(); // Start visualizing the pitch

//             // Start recording using MediaRecorder
//             mediaRecorderRef.current = new MediaRecorder(stream);

//             mediaRecorderRef.current.ondataavailable = (event) => {
//                 if (event.data.size > 0) {
//                     const blob = new Blob([event.data], { type: 'audio/wav' });

//                     // Send the audio chunk to the backend for emotion prediction
//                     processAudioPrediction(blob);
//                 }
//             };

//             mediaRecorderRef.current.start();
//             console.log('MediaRecorder started');

//             // Set interval to split the recording into chunks based on the selected interval
//             intervalRef.current = setInterval(() => {
//                 if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
//                     mediaRecorderRef.current.stop(); // Stop current recording
//                     mediaRecorderRef.current.start(); // Immediately start a new recording
//                 }
//             }, interval * 60 * 1000); // Convert minutes to milliseconds

//             setIsRecording(true);
//             chrome.runtime.sendMessage({ action: 'startRecording' });
//         } catch (error) {
//             console.error('Error accessing microphone', error);
//         }
//     };

//     const stopRecording = () => {
//         clearInterval(intervalRef.current); // Clear the interval
//         isRecordingActive.current = false; // Stop the visualization

//         if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
//             mediaRecorderRef.current.stop(); // Stop the final recording
//         }

//         setIsRecording(false);
//         setFeedback('Recording stopped.'); // Update feedback to "Recording stopped."

//         // Close the AudioContext and release microphone stream
//         if (audioContextRef.current) {
//             audioContextRef.current.close();
//             audioContextRef.current = null; // Reset the AudioContext reference
//         }

//         if (streamRef.current) {
//             streamRef.current.getTracks().forEach(track => track.stop()); // Stop all media tracks
//             streamRef.current = null;
//         }

//         chrome.runtime.sendMessage({ action: 'stopRecording' });
//     };

//     const processAudioPrediction = async (blob) => {
//         const formData = new FormData();
//         formData.append('file', blob, 'recording.wav');

//         try {
//             const response = await axios.post('http://localhost:5000/predict_emotion', formData, {
//                 headers: { 'Content-Type': 'multipart/form-data' },
//                 withCredentials: true
//             });

//             const detectedEmotion = response.data.emotion;
//             const emotionDetails = response.data.reason;

//             setEmotion(`Detected Emotion: ${detectedEmotion === 'stressed' ? 'Stressed' : 'Not Stressed'}`);
//             setDetails(`Details: ${emotionDetails}`);

//             // Fetch historical emotions after new prediction is made
//             fetchUserEmotions();

//             if (detectedEmotion === 'stressed') {
//                 const event = new CustomEvent('stressAlert', { detail: 'stressed' });
//                 window.dispatchEvent(event);
//                 chrome.runtime.sendMessage({ action: 'stressDetected' });
//             } else {
//                 chrome.runtime.sendMessage({ action: 'clearBadge' });
//             }

//             // Second API request (which I didn't remove)
//             const secondApiResponse = await axios.post('http://localhost:5000/another_endpoint', formData, {
//                 headers: { 'Content-Type': 'multipart/form-data' },
//                 withCredentials: true
//             });
//             console.log('Second API response:', secondApiResponse.data);

//         } catch (error) {
//             console.error('Error uploading the file', error);
//         }
//     };

//     const handleFilterChange = (event) => {
//         setFilter(event.target.value);
//     };

//     const handleIntervalChange = (e) => {
//         setIntervalTime(Number(e.target.value)); // Set interval in minutes
//     };

//     const filteredEmotions = userEmotions.filter(entry => {
//         if (filter === 'All') return true;
//         if (filter === 'Stressed') return entry.emotion === 'stressed';
//         if (filter === 'Not Stressed') return entry.emotion === 'not stressed';
//         return false;
//     });

//     return (
//         <>
//             <div className="dropdown-container">
//                 <div>
//                     <label className="dropdown-label">Select Interval (in minutes):</label>
//                     <input type="number" min="1" defaultValue="1" placeholder='Default value is 1 Minute' onChange={handleIntervalChange} className="select-dropdown" />
//                 </div>
//             </div>
//             <div className="btn-container">
//                 <button className="success" onClick={startRecording} disabled={isRecording}>Start</button>
//                 <button className="danger" onClick={stopRecording} disabled={!isRecording}>Stop</button>
//             </div>
//             <canvas id="pitchCanvas" width="600" height="100" style={{ border: '1px solid #007bff', marginTop: '20px', backgroundColor: '#3A4A6B' }}></canvas> {/* Pitch visual canvas */}
//             <div>
//                 {emotion && <p className="emotion">{emotion}</p>}
//                 {details && <p className="details">{details}</p>}
//                 {feedback && (
//                     <p className={`feedback ${!isRecording ? 'feedback-stopped' : ''}`}>
//                         {feedback}
//                     </p>
//                 )}
//             </div>
//             <div className="filter-container">
//                 <label className="filter-label">Filter Results:</label>
//                 <select value={filter} onChange={handleFilterChange} className="select-dropdown">
//                     <option value="All">All</option>
//                     <option value="Stressed">Stressed</option>
//                     <option value="Not Stressed">Not Stressed</option>
//                 </select>
//             </div>
//             <div className="emotion">
//                 <h2>Previous Results</h2>
//                 <div className="results-container">
//                     <ul>
//                         {filteredEmotions.map((entry, index) => (
//                             <li key={index}>
//                                 {new Date(entry.timestamp).toLocaleString()}: {entry.emotion === 'stressed' ? 'Stressed' : 'Not Stressed'}
//                             </li>
//                         ))}
//                     </ul>
//                 </div>
//             </div>
//         </>
//     );
// };

// export default Detector;
/* import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './detector.css';

const Detector = ({ username }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [emotion, setEmotion] = useState('');
    const [details, setDetails] = useState('');
    const [userEmotions, setUserEmotions] = useState([]);
    const [filter, setFilter] = useState('All');
    const [interval, setIntervalTime] = useState(1);
    const [feedback, setFeedback] = useState('');

    const mediaRecorderRef = useRef(null);
    const intervalRef = useRef(null);
    const isRecordingActive = useRef(false);
    const audioContextRef = useRef(null);
    const streamRef = useRef(null);

    useEffect(() => {
        fetchUserEmotions();
    }, []);

    const fetchUserEmotions = async () => {
        try {
            const response = await axios.get('http://localhost:5000/user_emotions', { withCredentials: true });
            setUserEmotions(response.data);
        } catch (error) {
            console.error("Error fetching user emotions:", error);
        }
    };

    const startRecording = async () => {
        setFeedback('Recording...');

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            }

            const analyser = audioContextRef.current.createAnalyser();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyser);

            const canvas = document.getElementById('pitchCanvas');
            const canvasContext = canvas.getContext('2d');

            analyser.fftSize = 2048;
            const bufferLength = analyser.fftSize;
            const dataArray = new Uint8Array(bufferLength);
            isRecordingActive.current = true;

            const drawPitch = () => {
                if (!isRecordingActive.current) return;
                requestAnimationFrame(drawPitch);

                analyser.getByteTimeDomainData(dataArray);
                canvasContext.fillStyle = '#3A4A6B';
                canvasContext.fillRect(0, 0, canvas.width, canvas.height);
                canvasContext.lineWidth = 2;
                canvasContext.strokeStyle = '#B0C4DE';
                canvasContext.beginPath();

                const sliceWidth = canvas.width / bufferLength;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = v * canvas.height / 2;
                    i === 0 ? canvasContext.moveTo(x, y) : canvasContext.lineTo(x, y);
                    x += sliceWidth;
                }

                canvasContext.lineTo(canvas.width, canvas.height / 2);
                canvasContext.stroke();
            };

            drawPitch();

            mediaRecorderRef.current = new MediaRecorder(stream);
            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    const blob = new Blob([event.data], { type: 'audio/wav' });
                    processAudioPrediction(blob);
                }
            };

            mediaRecorderRef.current.start();
            intervalRef.current = setInterval(() => {
                if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
                    mediaRecorderRef.current.stop();
                    mediaRecorderRef.current.start();
                }
            }, interval * 60 * 1000);

            setIsRecording(true);
        } catch (error) {
            console.error('Error accessing microphone', error);
        }
    };

    const stopRecording = () => {
        clearInterval(intervalRef.current);
        isRecordingActive.current = false;

        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }

        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        setIsRecording(false);
        setFeedback('Recording stopped.');
    };

    const processAudioPrediction = async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'recording.wav');

        try {
            const response = await axios.post('http://localhost:5000/predict_emotion', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                withCredentials: true
            });

            const detectedEmotion = response.data.emotion;
            const emotionDetails = response.data.reason;

            setEmotion(`Detected Emotion: ${detectedEmotion === 'stressed' ? 'Stressed' : 'Not Stressed'}`);
            setDetails(`Details: ${emotionDetails}`);
            fetchUserEmotions();

            // Optional: trigger UI-only alerts based on stress
            if (detectedEmotion === 'stressed') {
                const event = new CustomEvent('stressAlert', { detail: 'stressed' });
                window.dispatchEvent(event);
            }

            const secondApiResponse = await axios.post('http://localhost:5000/another_endpoint', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                withCredentials: true
            });

            console.log('Second API response:', secondApiResponse.data);

        } catch (error) {
            console.error('Error uploading the file', error);
        }
    };

    const handleFilterChange = (event) => setFilter(event.target.value);
    const handleIntervalChange = (e) => setIntervalTime(Number(e.target.value));

    const filteredEmotions = userEmotions.filter(entry => {
        if (filter === 'All') return true;
        return entry.emotion === filter.toLowerCase();
    });

    return (
        <>
            <div className="dropdown-container">
                <label className="dropdown-label">Select Interval (in minutes):</label>
                <input
                    type="number"
                    min="1"
                    defaultValue="1"
                    onChange={handleIntervalChange}
                    className="select-dropdown"
                />
            </div>

            <div className="btn-container">
                <button className="success" onClick={startRecording} disabled={isRecording}>Start</button>
                <button className="danger" onClick={stopRecording} disabled={!isRecording}>Stop</button>
            </div>

            <canvas id="pitchCanvas" width="600" height="100"
                style={{ border: '1px solid #007bff', marginTop: '20px', backgroundColor: '#3A4A6B' }}></canvas>

            <div>
                {emotion && <p className="emotion">{emotion}</p>}
                {details && <p className="details">{details}</p>}
                {feedback && <p className={`feedback ${!isRecording ? 'feedback-stopped' : ''}`}>{feedback}</p>}
            </div>

            <div className="filter-container">
                <label className="filter-label">Filter Results:</label>
                <select value={filter} onChange={handleFilterChange} className="select-dropdown">
                    <option value="All">All</option>
                    <option value="Stressed">Stressed</option>
                    <option value="Not Stressed">Not Stressed</option>
                </select>
            </div>

            <div className="emotion">
                <h2>Previous Results</h2>
                <div className="results-container">
                    <ul>
                        {filteredEmotions.map((entry, index) => (
                            <li key={index}>
                                {new Date(entry.timestamp).toLocaleString()}: {entry.emotion === 'stressed' ? 'Stressed' : 'Not Stressed'}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </>
    );
};

export default Detector;
 */
/* import React, { useState, useRef, useEffect, useMemo } from "react";
import axios from "axios";
import "./detector.css";

const Detector = ({ username }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [emotionLabel, setEmotionLabel] = useState("");
  const [details, setDetails] = useState("");
  const [userEmotions, setUserEmotions] = useState([]);
  const [filter, setFilter] = useState("All"); // All | Stressed | Not Stressed
  const [intervalMin, setIntervalMin] = useState(1);
  const [feedback, setFeedback] = useState("");

  const mediaRecorderRef = useRef(null);
  const intervalRef = useRef(null);
  const isRecordingActive = useRef(false);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  // --- helpers ---
  const parseTimestamp = (ts) => {
    if (!ts) return "";
    // Mongo can return ISO string or {$date: "..."}
    const raw = typeof ts === "string" ? ts : ts.$date || ts;
    return new Date(raw);
  };

  const loadUserEmotions = async () => {
    try {
      const res = await axios.get("http://localhost:5000/user_emotions", {
        withCredentials: true,
      });
      const data = Array.isArray(res.data) ? res.data : [];
      // sort newest -> oldest by timestamp
      data.sort((a, b) => {
        const ta = parseTimestamp(a.timestamp)?.getTime() || 0;
        const tb = parseTimestamp(b.timestamp)?.getTime() || 0;
        return tb - ta;
      });
      setUserEmotions(data);
    } catch (err) {
      console.error("Error fetching user emotions:", err);
      if (err?.response?.status === 401) {
        setFeedback("Please sign in to save and view results.");
      }
    }
  };

  useEffect(() => {
    loadUserEmotions();
  }, []);

  // Filtered view
  const filteredEmotions = useMemo(() => {
    if (filter === "All") return userEmotions;
    const want = filter.toLowerCase(); // "stressed" | "not stressed"
    return userEmotions.filter((e) => (e.emotion || "").toLowerCase() === want);
  }, [filter, userEmotions]);

  // --- Recording UI / waveform ---
  const drawWave = (analyser, dataArray, bufferLength, ctx, canvas) => {
    if (!isRecordingActive.current) return;
    requestAnimationFrame(() =>
      drawWave(analyser, dataArray, bufferLength, ctx, canvas)
    );

    analyser.getByteTimeDomainData(dataArray);
    ctx.fillStyle = "#3A4A6B";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#B0C4DE";
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  };

  const startRecording = async () => {
    setFeedback("Recording...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      if (!audioContextRef.current) {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioContextRef.current = new AC();
      }

      const analyser = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyser);

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      analyser.fftSize = 2048;
      const bufferLength = analyser.fftSize;
      const dataArray = new Uint8Array(bufferLength);
      isRecordingActive.current = true;
      drawWave(analyser, dataArray, bufferLength, ctx, canvas);

      // Prepare MediaRecorder
      const mimeType =
        MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm";
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          processAudioPrediction(event.data);
        }
      };

      mediaRecorderRef.current.start();

      // chunk at chosen interval
      intervalRef.current = setInterval(() => {
        if (
          mediaRecorderRef.current &&
          mediaRecorderRef.current.state !== "inactive"
        ) {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.start();
        }
      }, Math.max(1, intervalMin) * 60 * 1000);

      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone", err);
      setFeedback("Microphone access failed.");
    }
  };

  const stopRecording = () => {
    clearInterval(intervalRef.current);
    isRecordingActive.current = false;

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    setIsRecording(false);
    setFeedback("Recording stopped.");
  };

  // Ensure cleanup on unmount
  useEffect(() => {
    return () => {
      try {
        stopRecording();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Prediction upload ---
  const processAudioPrediction = async (blob) => {
    // Convert webm blob to a generic file name (backend re-exports to wav)
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");

    try {
      const res = await axios.post("http://localhost:5000/predict_emotion", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        withCredentials: true,
      });

      const detectedEmotion = res.data.emotion; // "stressed" | "not stressed"
      const reason = res.data.reason || "Model classified the emotion.";

      setEmotionLabel(
        `Detected Emotion: ${detectedEmotion === "stressed" ? "Stressed" : "Not Stressed"}`
      );
      setDetails(`Details: ${reason}`);

      // Refresh history list
      await loadUserEmotions();

      // Dispatch stress alert for the chatbot speech, if needed
      if (detectedEmotion === "stressed") {
        const event = new CustomEvent("stressAlert", { detail: "stressed" });
        window.dispatchEvent(event);
      }
    } catch (err) {
      console.error("Error uploading for prediction:", err);
      setFeedback("Upload failed. Please try again.");
    }
  };

  return (
    <>
      <div className="dropdown-container">
        <label className="dropdown-label">Select Interval (in minutes):</label>
        <input
          type="number"
          min="1"
          value={intervalMin}
          onChange={(e) => setIntervalMin(Number(e.target.value) || 1)}
          className="select-dropdown"
        />
      </div>

      <div className="btn-container">
        <button className="success" onClick={startRecording} disabled={isRecording}>
          Start
        </button>
        <button className="danger" onClick={stopRecording} disabled={!isRecording}>
          Stop
        </button>
      </div>

      <canvas
        ref={canvasRef}
        id="pitchCanvas"
        width="600"
        height="100"
        style={{ border: "1px solid #007bff", marginTop: 20, backgroundColor: "#3A4A6B" }}
      />

      <div>
        {emotionLabel && <p className="emotion">{emotionLabel}</p>}
        {details && <p className="details">{details}</p>}
        {feedback && (
          <p className={`feedback ${!isRecording ? "feedback-stopped" : ""}`}>{feedback}</p>
        )}
      </div>

      <div className="filter-container">
        <label className="filter-label">Filter Results:</label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="select-dropdown"
        >
          <option value="All">All</option>
          <option value="Stressed">Stressed</option>
          <option value="Not Stressed">Not Stressed</option>
        </select>
      </div>

      <div className="emotion">
        <h2>Previous Results</h2>
        <div className="results-container">
          {filteredEmotions.length === 0 ? (
            <p style={{ opacity: 0.7 }}>
              No results yet. Start a recording to see your history here.
            </p>
          ) : (
            <ul>
              {filteredEmotions.map((entry, idx) => {
                const ts = parseTimestamp(entry.timestamp);
                const when = ts ? ts.toLocaleString() : "Unknown time";
                const label =
                  (entry.emotion || "").toLowerCase() === "stressed"
                    ? "Stressed"
                    : "Not Stressed";
                return (
                  <li key={idx}>
                    {when}: {label}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </>
  );
};

export default Detector;
 */

// import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
// import axios from "axios";
// import "./detector.css";

// const Detector = ({ username }) => {
//   const [isRecording, setIsRecording] = useState(false);
//   const [emotionLabel, setEmotionLabel] = useState("");
//   const [details, setDetails] = useState("");
//   const [userEmotions, setUserEmotions] = useState([]);
//   const [filter, setFilter] = useState("All"); // All | Stressed | Not Stressed
//   const [intervalMin, setIntervalMin] = useState(1);
//   const [feedback, setFeedback] = useState("");

//   // Random intervention modal
//   const [showIntervention, setShowIntervention] = useState(false);
//   const [interventionType, setInterventionType] = useState(null);

//   const mediaRecorderRef = useRef(null);
//   const isRecordingActive = useRef(false);
//   const audioContextRef = useRef(null);
//   const streamRef = useRef(null);
//   const canvasRef = useRef(null);

//   // ⭐ timers for manual chunking (requestData) + watchdog
//   const sendTimerRef = useRef(null);
//   const watchdogRef = useRef(null);
//   const intervalRef = useRef(null);

//   const lastChunkAtRef = useRef(0);

//   // draw waveform (unchanged)
//   const drawWave = (analyser, dataArray, bufferLength, ctx, canvas) => {
//     if (!isRecordingActive.current) return;
//     requestAnimationFrame(() => drawWave(analyser, dataArray, bufferLength, ctx, canvas));
//     analyser.getByteTimeDomainData(dataArray);
//     ctx.fillStyle = "#3A4A6B";
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
//     ctx.lineWidth = 2;
//     ctx.strokeStyle = "#B0C4DE";
//     ctx.beginPath();
//     const sliceWidth = canvas.width / bufferLength;
//     let x = 0;
//     for (let i = 0; i < bufferLength; i++) {
//       const v = dataArray[i] / 128.0;
//       const y = (v * canvas.height) / 2;
//       i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
//       x += sliceWidth;
//     }
//     ctx.lineTo(canvas.width, canvas.height / 2);
//     ctx.stroke();
//   };

//   // --- history helpers ---
//   const parseTimestamp = (ts) => {
//     const raw = typeof ts === "string" ? ts : ts?.$date || ts || "";
//     return raw ? new Date(raw) : null;
//   };

//   const loadUserEmotions = useCallback(async () => {
//     try {
//       const res = await axios.get("http://localhost:5000/user_emotions", {
//         withCredentials: true,
//       });
//       const data = Array.isArray(res.data) ? res.data : [];
//       data.sort((a, b) => {
//         const ta = parseTimestamp(a.timestamp)?.getTime() || 0;
//         const tb = parseTimestamp(b.timestamp)?.getTime() || 0;
//         return tb - ta;
//       });
//       setUserEmotions(data);
//     } catch (err) {
//       console.error("Error fetching user emotions:", err);
//       if (err?.response?.status === 401) setFeedback("Please sign in to save and view results.");
//     }
//   }, []);

//   useEffect(() => {
//     loadUserEmotions();
//   }, [loadUserEmotions]);

//   const filteredEmotions = useMemo(() => {
//     if (filter === "All") return userEmotions;
//     const want = filter.toLowerCase();
//     return userEmotions.filter((e) => (e.emotion || "").toLowerCase() === want);
//   }, [filter, userEmotions]);

//   // --- upload each chunk for prediction ---
// const processAudioPrediction = async (blob) => {
//   // pick the right extension for ffmpeg to guess format reliably
//   const type = blob.type || "";
//   const ext = /ogg/.test(type) ? "ogg" : "webm";
//   const file = new File([blob], `clip.${ext}`, { type: type || `audio/${ext}` });

//   const fd = new FormData();
//   fd.append("file", file); // field name MUST be "file"

//   try {
//     const res = await axios.post("http://localhost:5000/predict_emotion", fd, {
//       withCredentials: true,
//       // ❌ remove this: headers: { "Content-Type": "multipart/form-data" }
//     });

//     const detected = res.data.emotion; // "stressed" | "not stressed"
//     const reason   = res.data.reason || "Model classified the emotion.";
//     setEmotionLabel(`Detected Emotion: ${detected === "stressed" ? "Stressed" : "Not Stressed"}`);
//     setDetails(`Details: ${reason}`);
//     await loadUserEmotions();

//     if (detected === "stressed") {
//       // your alert / intervention logic
//       const evt = new CustomEvent("stressAlert", { detail: "stressed" });
//       window.dispatchEvent(evt);
//     }
//   } catch (err) {
//     console.error("Upload error", err.response?.status, err.response?.data || err.message);
//     setFeedback("Upload failed. Try again.");
//   }
// };

//   // --- Start recorder using manual requestData loop (more reliable than start(timeslice)) ---
//   const startMediaRecorder = useCallback(
//     async (timesliceMs) => {
//       const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//       streamRef.current = stream;

//       if (!audioContextRef.current) {
//         const AC = window.AudioContext || window.webkitAudioContext;
//         audioContextRef.current = new AC();
//       }

//       // waveform
//       const analyser = audioContextRef.current.createAnalyser();
//       const source = audioContextRef.current.createMediaStreamSource(stream);
//       source.connect(analyser);
//       const canvas = canvasRef.current;
//       const ctx = canvas.getContext("2d");
//       analyser.fftSize = 2048;
//       const bufferLength = analyser.fftSize;
//       const dataArray = new Uint8Array(bufferLength);
//       isRecordingActive.current = true;
//       drawWave(analyser, dataArray, bufferLength, ctx, canvas);

//       // pick a supported mime
//       const preferred = [
//         "audio/webm;codecs=opus",
//         "audio/webm",
//         "audio/ogg;codecs=opus",
//         "audio/ogg",
//       ];
//       const mimeType = preferred.find((m) => MediaRecorder.isTypeSupported(m)) || "";
//       const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
//       mediaRecorderRef.current = mr;

//       mr.ondataavailable = (event) => {
//         if (event.data && event.data.size > 0) {
//           lastChunkAtRef.current = Date.now(); // ⭐ heartbeat for watchdog
//           processAudioPrediction(event.data);
//         }
//       };

//       mr.start(); // ⭐ no timeslice here

//       // ⭐ Kick out the first small sample early so UI shows results fast
//       setTimeout(() => {
//         try { mr.requestData(); } catch {}
//       }, 5000);

//       // ⭐ Main periodic chunking
//       const period = Math.max(60_000, Math.max(1, timesliceMs)); // >= 1 minute in ms (you pass minutes*60*1000 already)
//       sendTimerRef.current = setInterval(() => {
//         try { mr.requestData(); } catch {}
//       }, period);

//       // ⭐ Watchdog: if no chunk arrives within 1.5×period, force a request
//       lastChunkAtRef.current = Date.now();
//       watchdogRef.current = setInterval(() => {
//         const now = Date.now();
//         if (now - lastChunkAtRef.current > period * 1.5) {
//           try { mr.requestData(); } catch {}
//           lastChunkAtRef.current = now;
//         }
//       }, Math.min(30_000, Math.max(5_000, period / 3))); // check every 5–30s
//     },
//     [processAudioPrediction]
//   );

//   const stopMediaRecorder = useCallback(() => {
//     isRecordingActive.current = false;

//     // stop timers
//     if (sendTimerRef.current) { clearInterval(sendTimerRef.current); sendTimerRef.current = null; }
//     if (watchdogRef.current) { clearInterval(watchdogRef.current); watchdogRef.current = null; }

//     // flush last chunk
//     try { mediaRecorderRef.current && mediaRecorderRef.current.requestData(); } catch {}

//     const mr = mediaRecorderRef.current;
//     if (mr && mr.state !== "inactive") {
//       try { mr.stop(); } catch {}
//     }
//     mediaRecorderRef.current = null;

//     if (audioContextRef.current) {
//       try { audioContextRef.current.close(); } catch {}
//       audioContextRef.current = null;
//     }

//     if (streamRef.current) {
//       try { streamRef.current.getTracks().forEach((t) => t.stop()); } catch {}
//       streamRef.current = null;
//     }
//   }, []);

//   // --- Public Start/Stop handlers ---
// const startRecording = async () => {
//   setFeedback("Recording…");

//   try {
//     const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//     streamRef.current = stream;

//     // (your waveform/analyser hookup stays the same)

//     // Pick a supported mime type so the blob has the right container/header
//     const options = {};
//     const preferred = [
//       "audio/webm;codecs=opus",
//       "audio/webm",
//       "audio/ogg;codecs=opus",
//       "audio/ogg",
//     ];
//     for (const m of preferred) {
//       if (MediaRecorder.isTypeSupported(m)) { options.mimeType = m; break; }
//     }

//     const mr = new MediaRecorder(stream, options);
//     mediaRecorderRef.current = mr;

//     // Upload each finalized chunk
//     mr.ondataavailable = async (e) => {
//       if (!e.data || e.data.size === 0) return;

//       // choose extension from the actual blob type (prevents ffmpeg “matroska/webm EBML header” errors)
//       const t = (e.data.type || "").toLowerCase();
//       const ext = t.includes("ogg") ? "ogg" : t.includes("webm") ? "webm" : "webm";

//       const formData = new FormData();
//       formData.append("file", e.data, `recording.${ext}`);

//       try {
//         const res = await axios.post("http://localhost:5000/predict_emotion", formData, {
//           withCredentials: true, // do NOT set Content-Type; browser sets the boundary
//         });
//         const { emotion, reason } = res.data;
//         setEmotionLabel(`Detected Emotion: ${emotion === "stressed" ? "Stressed" : "Not Stressed"}`);
//         setDetails(`Details: ${reason || "Model classified the emotion."}`);
//         await loadUserEmotions();
//       } catch (err) {
//         console.error("Upload error", err);
//         setFeedback("Upload failed.");
//       }
//     };

//     // When a chunk finishes, immediately start a new one
//     mr.onstop = () => {
//       if (isRecordingActive.current) {
//         try { mr.start(); } catch {}
//       }
//     };

//     // Start the first recording (no timeslice so the browser writes a valid header)
//     isRecordingActive.current = true;
//     mr.start();

//     // Every interval: stop() to finalize a valid file (fires ondataavailable), then onstop restarts
//     const periodMs = Math.max(1, intervalMin) * 60 * 1000;
//     intervalRef.current = setInterval(() => {
//       if (mr.state === "recording") {
//         try { mr.stop(); } catch {}
//       }
//     }, periodMs);

//     setIsRecording(true);
//   } catch (err) {
//     console.error("Mic access failed:", err);
//     setFeedback("Microphone access failed.");
//   }
// };

// const stopRecording = () => {
//   if (intervalRef.current) {
//     clearInterval(intervalRef.current);
//     intervalRef.current = null;
//   }
//   isRecordingActive.current = false;

//   const mr = mediaRecorderRef.current;
//   if (mr && mr.state === "recording") {
//     try { mr.stop(); } catch {}
//   }
//   mediaRecorderRef.current = null;

//   // Close audio context & release mic
//   if (audioContextRef.current) {
//     try { audioContextRef.current.close(); } catch {}
//     audioContextRef.current = null;
//   }
//   if (streamRef.current) {
//     try { streamRef.current.getTracks().forEach(t => t.stop()); } catch {}
//     streamRef.current = null;
//   }

//   setIsRecording(false);
//   setFeedback("Recording stopped.");
// };


//   // If user changes the interval while recording, restart recorder with new period
//   useEffect(() => {
//     if (!isRecording) return;
//     const apply = async () => {
//       stopMediaRecorder();
//       try {
//         await startMediaRecorder(Math.max(1, intervalMin) * 60 * 1000);
//       } catch (e) {
//         console.error("Failed to apply new interval:", e);
//         setFeedback("Failed to apply new interval.");
//       }
//     };
//     apply();
//     // eslint-disable-next-line react-hooks/exhaustive-deps
//   }, [intervalMin]);

//   // Cleanup on unmount
//   useEffect(() => {
//     return () => {
//       try { stopMediaRecorder(); } catch {}
//     };
//   }, [stopMediaRecorder]);

//   // --------- Random Intervention ----------
//   const speak = (text) => {
//     if (!("speechSynthesis" in window)) return;
//     const u = new SpeechSynthesisUtterance(text);
//     u.rate = 1;
//     u.pitch = 1;
//     u.volume = 1;
//     try { window.speechSynthesis.cancel(); } catch {}
//     try { window.speechSynthesis.speak(u); } catch {}
//   };

//   const showRandomIntervention = () => {
//     const options = ["breathing", "break", "dietVoice", "chatbot"];
//     const pick = options[Math.floor(Math.random() * options.length)];
//     setInterventionType(pick);
//     setShowIntervention(true);
//     if (pick === "dietVoice") {
//       const lines = [
//         "Try a rainbow bowl today. Blueberries, oranges, spinach, carrots, and avocado. Hydrate with water or herbal tea.",
//         "Snack idea: apple slices with a handful of walnuts. For dinner, mixed greens, cherry tomatoes, cucumber, and olive oil.",
//         "Keep it simple: banana, a small bowl of oats, and a side salad with leafy greens.",
//       ];
//       const tip = lines[Math.floor(Math.random() * lines.length)];
//       speak(`Here is a quick stress friendly tip: ${tip}`);
//     }
//     if (pick === "chatbot") {
//       const evt = new CustomEvent("stressAlert", { detail: "stressed" });
//       window.dispatchEvent(evt);
//     }
//   };

//   // Breathing coach state (unchanged)
//   const INHALE_MS = 30000;
//   const EXHALE_MS = 30000;
//   const [breathingRunning, setBreathingRunning] = useState(false);
//   const [breathingPhase, setBreathingPhase] = useState("idle");
//   const [breathingProgress, setBreathingProgress] = useState(0);
//   const breathingRunningRef = useRef(false);
//   const breathingIntervalRef = useRef(null);
//   const breathingPhaseTimerRef = useRef(null);

//   const stopBreathing = useCallback(() => {
//     breathingRunningRef.current = false;
//     setBreathingRunning(false);
//     setBreathingPhase("idle");
//     setBreathingProgress(0);
//     if (breathingIntervalRef.current) { clearInterval(breathingIntervalRef.current); breathingIntervalRef.current = null; }
//     if (breathingPhaseTimerRef.current) { clearTimeout(breathingPhaseTimerRef.current); breathingPhaseTimerRef.current = null; }
//   }, []);

//   const runOneBreathingCycle = useCallback(() => {
//     if (!breathingRunningRef.current) return;
//     setBreathingPhase("inhale");
//     setBreathingProgress(0);
//     speak("Breathe in slowly.");
//     const tick = 50;
//     const startIn = Date.now();
//     if (breathingIntervalRef.current) clearInterval(breathingIntervalRef.current);
//     breathingIntervalRef.current = setInterval(() => {
//       const p = Math.min(1, (Date.now() - startIn) / INHALE_MS);
//       setBreathingProgress(p);
//       if (p >= 1) clearInterval(breathingIntervalRef.current);
//     }, tick);

//     if (breathingPhaseTimerRef.current) clearTimeout(breathingPhaseTimerRef.current);
//     breathingPhaseTimerRef.current = setTimeout(() => {
//       if (!breathingRunningRef.current) return;
//       setBreathingPhase("exhale");
//       setBreathingProgress(1);
//       speak("Now exhale gently.");
//       const startEx = Date.now();
//       if (breathingIntervalRef.current) clearInterval(breathingIntervalRef.current);
//       breathingIntervalRef.current = setInterval(() => {
//         const p = 1 - Math.min(1, (Date.now() - startEx) / EXHALE_MS);
//         setBreathingProgress(p);
//         if (p <= 0) clearInterval(breathingIntervalRef.current);
//       }, tick);
//       if (breathingPhaseTimerRef.current) clearTimeout(breathingPhaseTimerRef.current);
//       breathingPhaseTimerRef.current = setTimeout(() => {
//         if (breathingRunningRef.current) runOneBreathingCycle();
//       }, EXHALE_MS);
//     }, INHALE_MS);
//   }, [EXHALE_MS, INHALE_MS]);

//   const startBreathing = useCallback(() => {
//     breathingRunningRef.current = true;
//     setBreathingRunning(true);
//     runOneBreathingCycle();
//   }, [runOneBreathingCycle]);

//   // Break coach state (unchanged)
//   const BREAK_TOTAL_SECONDS = 5 * 60;
//   const [breakRunning, setBreakRunning] = useState(false);
//   const [breakSecondsLeft, setBreakSecondsLeft] = useState(BREAK_TOTAL_SECONDS);
//   const [paceUp, setPaceUp] = useState(true);
//   const breakTickRef = useRef(null);

//   const startBreak = useCallback(() => {
//     setBreakSecondsLeft(BREAK_TOTAL_SECONDS);
//     setPaceUp(true);
//     setBreakRunning(true);
//     speak("Let’s take a five minute break. Start walking. We will alternate pace up and pace down every thirty seconds.");
//     if (breakTickRef.current) clearInterval(breakTickRef.current);
//     breakTickRef.current = setInterval(() => {
//       setBreakSecondsLeft((s) => {
//         const next = Math.max(0, s - 1);
//         const elapsed = BREAK_TOTAL_SECONDS - next;
//         if (elapsed > 0 && elapsed % 30 === 0) {
//           setPaceUp((prev) => {
//             const nu = !prev;
//             speak(nu ? "Pace up." : "Pace down.");
//             return nu;
//           });
//         }
//         if (next === 0) {
//           clearInterval(breakTickRef.current);
//           breakTickRef.current = null;
//           setBreakRunning(false);
//           speak("Break complete. Great job!");
//         }
//         return next;
//       });
//     }, 1000);
//   }, []);

//   const stopBreak = useCallback(() => {
//     setBreakRunning(false);
//     setBreakSecondsLeft(BREAK_TOTAL_SECONDS);
//     setPaceUp(true);
//     if (breakTickRef.current) {
//       clearInterval(breakTickRef.current);
//       breakTickRef.current = null;
//     }
//   }, []);

//   const closeIntervention = () => {
//     if (interventionType === "breathing") stopBreathing();
//     if (interventionType === "break") stopBreak();
//     setShowIntervention(false);
//     setInterventionType(null);
//   };

//   // sizes
//   const minSize = 80;
//   const maxSize = 220;
//   const circleSize = Math.round(minSize + (maxSize - minSize) * breathingProgress);
//   const mm = String(Math.floor(breakSecondsLeft / 60)).padStart(2, "0");
//   const ss = String(breakSecondsLeft % 60).padStart(2, "0");

//   return (
//     <>
//       <div className="dropdown-container">
//         <label className="dropdown-label">Select Interval (in minutes):</label>
//         <input
//           type="number"
//           min="1"
//           value={intervalMin}
//           onChange={(e) => setIntervalMin(Number(e.target.value) || 1)}
//           className="select-dropdown"
//         />
//       </div>

//       <div className="btn-container">
//         <button className="success" onClick={startRecording} disabled={isRecording}>
//           Start
//         </button>
//         <button className="danger" onClick={stopRecording} disabled={!isRecording}>
//           Stop
//         </button>
//       </div>

//       <canvas
//         ref={canvasRef}
//         id="pitchCanvas"
//         width="600"
//         height="100"
//         style={{ border: "1px solid #007bff", marginTop: 20, backgroundColor: "#3A4A6B" }}
//       />

//       <div>
//         {emotionLabel && <p className="emotion">{emotionLabel}</p>}
//         {details && <p className="details">{details}</p>}
//         {feedback && (
//           <p className={`feedback ${!isRecording ? "feedback-stopped" : ""}`}>{feedback}</p>
//         )}
//       </div>

//       <div className="filter-container">
//         <label className="filter-label">Filter Results:</label>
//         <select
//           value={filter}
//           onChange={(e) => setFilter(e.target.value)}
//           className="select-dropdown"
//         >
//           <option value="All">All</option>
//           <option value="Stressed">Stressed</option>
//           <option value="Not Stressed">Not Stressed</option>
//         </select>
//       </div>

//       <div className="emotion">
//         <h2>Previous Results</h2>
//         <div className="results-container">
//           {filteredEmotions.length === 0 ? (
//             <p style={{ opacity: 0.7 }}>
//               No results yet. Start a recording to see your history here.
//             </p>
//           ) : (
//             <ul>
//               {filteredEmotions.map((entry, idx) => {
//                 const ts = parseTimestamp(entry.timestamp);
//                 const when = ts ? ts.toLocaleString() : "Unknown time";
//                 const label =
//                   (entry.emotion || "").toLowerCase() === "stressed"
//                     ? "Stressed"
//                     : "Not Stressed";
//                 return (
//                   <li key={idx}>
//                     {when}: {label}
//                   </li>
//                 );
//               })}
//             </ul>
//           )}
//         </div>
//       </div>

//       {/* Random Intervention Modal */}
//       {showIntervention && (
//         <div className="bc-overlay show">
//           <div className="bc-modal">
//             <div className="bc-title">Quick Support</div>

//             {interventionType === "breathing" && (
//               <div className="bc-stage">
//                 <div className="bc-circle" style={{ width: circleSize, height: circleSize }} />
//                 <div className="bc-instructions">
//                   {breathingPhase === "inhale" && "Breathe in…"}
//                   {breathingPhase === "exhale" && "Breathe out…"}
//                   {breathingPhase === "idle" && "Ready to begin guided breathing?"}
//                 </div>
//                 <div className="bc-actions">
//                   {!breathingRunning ? (
//                     <button className="bc-btn bc-start" onClick={startBreathing}>Start</button>
//                   ) : (
//                     <button className="bc-btn bc-stop" onClick={stopBreathing}>Stop</button>
//                   )}
//                   <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
//                 </div>
//               </div>
//             )}

//             {interventionType === "break" && (
//               <div className="bc-stage">
//                 <div className="break-timer">{mm}:{ss}</div>
//                 <div className="bc-instructions" style={{ textAlign: "center" }}>
//                   {breakRunning
//                     ? (paceUp ? "Pace up. Walk a bit faster." : "Pace down. Slow your steps.")
//                     : "Press Start to begin a 5-minute walking break."}
//                 </div>
//                 <div className="bc-actions">
//                   {!breakRunning ? (
//                     <button className="bc-btn bc-start" onClick={startBreak}>Start</button>
//                   ) : (
//                     <button className="bc-btn bc-stop" onClick={stopBreak}>Stop</button>
//                   )}
//                   <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
//                 </div>
//               </div>
//             )}

//             {interventionType === "dietVoice" && (
//               <div className="bc-stage">
//                 <div className="bc-instructions" style={{ textAlign: "center" }}>
//                   Playing a short voice tip on fruits and vegetables for stress…
//                 </div>
//                 <div className="bc-actions">
//                   <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
//                 </div>
//               </div>
//             )}

//             {interventionType === "chatbot" && (
//               <div className="bc-stage">
//                 <div className="bc-instructions" style={{ textAlign: "center" }}>
//                   Open the Stress Bot to chat right away.
//                 </div>
//                 <div className="bc-actions">
//                   <button
//                     className="bc-btn bc-start"
//                     onClick={() => {
//                       const evt = new CustomEvent("stressAlert", { detail: "stressed" });
//                       window.dispatchEvent(evt);
//                       closeIntervention();
//                     }}
//                   >
//                     Open Stress Bot
//                   </button>
//                   <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
//                 </div>
//               </div>
//             )}
//           </div>

//           <style>{`
//             .bc-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.55); display: none; align-items: center; justify-content: center; z-index: 9999; }
//             .bc-overlay.show { display: flex; }
//             .bc-modal { width: 90%; max-width: 520px; background: #51558b; color: #fff; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
//             .bc-title { text-align: center; font-weight: 700; font-size: 20px; margin-bottom: 12px; }
//             .bc-stage { display: grid; place-items: center; gap: 12px; padding: 12px; }
//             .bc-circle { border-radius: 50%; background: radial-gradient(circle at 30% 30%, #93a5ff, #6b73ff); transition: width 50ms linear, height 50ms linear; box-shadow: 0 6px 18px rgba(0,0,0,0.25) inset, 0 8px 22px rgba(0,0,0,0.2); }
//             .bc-instructions { font-size: 18px; font-weight: 600; text-align:center; }
//             .bc-actions { display: flex; gap: 10px; justify-content: center; margin-top: 12px; }
//             .bc-btn { border: 0; padding: 10px 16px; border-radius: 999px; cursor: pointer; font-weight: 600; }
//             .bc-start { background: #2ecc71; color: #08321b; }
//             .bc-stop  { background: #ffcc66; color: #3a2b00; }
//             .bc-close { background: #e25757; color: #fff; }
//             .break-timer { font-size: 28px; font-weight: 700; letter-spacing: 1px; }
//           `}</style>
//         </div>
//       )}
//     </>
//   );
// };

// export default Detector;
import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import axios from "axios";
import "./detector.css";

const Detector = ({ username }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [emotionLabel, setEmotionLabel] = useState("");
  const [details, setDetails] = useState("");
  const [userEmotions, setUserEmotions] = useState([]);
  const [filter, setFilter] = useState("All"); // All | Stressed | Not Stressed
  const [intervalMin, setIntervalMin] = useState(1);
  const [feedback, setFeedback] = useState("");

  // interventions
  const [showIntervention, setShowIntervention] = useState(false);
  const [interventionType, setInterventionType] = useState(null);
  const lastInterventionAtRef = useRef(0);

  // media + timers
  const mediaRecorderRef = useRef(null);
  const intervalRef = useRef(null);
  const isRecordingActive = useRef(false);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  // -------- helpers --------
  const parseTimestamp = (ts) => {
    const raw = typeof ts === "string" ? ts : ts?.$date || ts || "";
    return raw ? new Date(raw) : null;
  };

  const loadUserEmotions = useCallback(async () => {
    try {
      const res = await axios.get("https://persuasive.research.cs.dal.ca/smsys/user_emotions", {
        withCredentials: true,
      });
      const data = Array.isArray(res.data) ? res.data : [];
      data.sort((a, b) => {
        const ta = parseTimestamp(a.timestamp)?.getTime() || 0;
        const tb = parseTimestamp(b.timestamp)?.getTime() || 0;
        return tb - ta;
      });
      setUserEmotions(data);
    } catch (err) {
      console.error("Error fetching user emotions:", err);
      if (err?.response?.status === 401) setFeedback("Please sign in to save and view results.");
    }
  }, []);

  useEffect(() => {
    loadUserEmotions();
  }, [loadUserEmotions]);

  const filteredEmotions = useMemo(() => {
    if (filter === "All") return userEmotions;
    const want = filter.toLowerCase();
    return userEmotions.filter((e) => (e.emotion || "").toLowerCase() === want);
  }, [filter, userEmotions]);

  // waveform draw
  const drawWave = (analyser, dataArray, bufferLength, ctx, canvas) => {
    if (!isRecordingActive.current) return;
    requestAnimationFrame(() => drawWave(analyser, dataArray, bufferLength, ctx, canvas));
    analyser.getByteTimeDomainData(dataArray);
    ctx.fillStyle = "#3A4A6B";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#B0C4DE";
    ctx.beginPath();
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  };

  // -------- upload a chunk --------
  const processAudioPrediction = async (blob) => {
    // ensure backend can sniff the container correctly
    const type = blob.type || "";
    const ext = /ogg/.test(type) ? "ogg" : "webm";
    const file = new File([blob], `clip.${ext}`, { type: type || `audio/${ext}` });

    const fd = new FormData();
    fd.append("file", file); // field must be "file"

    try {
      const res = await axios.post("https://persuasive.research.cs.dal.ca/smsys/predict_emotion", fd, {
        withCredentials: true,
        // don't set Content-Type; the browser will add the correct boundary
      });

      const { emotion, reason } = res.data; // "stressed" | "not stressed"
      setEmotionLabel(`Detected Emotion: ${emotion === "stressed" ? "Stressed" : "Not Stressed"}`);
      setDetails(`Details: ${reason || "Model classified the emotion."}`);

      await loadUserEmotions();

      if (emotion === "stressed") {
        const now = Date.now();
        const COOLDOWN_MS = 2 * 60 * 1000; // avoid spamming prompts
        if (now - (lastInterventionAtRef.current || 0) > COOLDOWN_MS) {
          lastInterventionAtRef.current = now;
          showRandomIntervention();
        }
      }
    } catch (err) {
      console.error("Upload error", err?.response?.status, err?.response?.data || err.message);
      setFeedback("Upload failed. Try again.");
    }
  };

  // -------- start/stop recording --------
  const startRecording = async () => {
    setFeedback("Recording...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // audio context + analyser for waveform
      if (!audioContextRef.current) {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioContextRef.current = new AC();
      }
      const analyser = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyser);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      analyser.fftSize = 2048;
      const bufferLength = analyser.fftSize;
      const dataArray = new Uint8Array(bufferLength);
      isRecordingActive.current = true;
      drawWave(analyser, dataArray, bufferLength, ctx, canvas);

      // pick a supported container/codec
      const options = {};
      const preferred = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
      ];
      for (const m of preferred) {
        if (MediaRecorder.isTypeSupported(m)) { options.mimeType = m; break; }
      }

      const mr = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) processAudioPrediction(e.data);
      };

      // when a chunk finalizes, immediately start the next one
      mr.onstop = () => {
        if (isRecordingActive.current) {
          try { mr.start(); } catch {}
        }
      };

      // start first recording (no timeslice → valid headers)
      mr.start();
      setIsRecording(true);

      // every N minutes, finalize the chunk (fires ondataavailable), then onstop restarts
      const periodMs = Math.max(1, intervalMin) * 60 * 1000;
      intervalRef.current = setInterval(() => {
        if (mr.state === "recording") {
          try { mr.stop(); } catch {}
        }
      }, periodMs);

    } catch (err) {
      console.error("Mic access failed:", err);
      setFeedback("Microphone access failed.");
    }
  };

  const stopRecording = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    isRecordingActive.current = false;

    const mr = mediaRecorderRef.current;
    if (mr && mr.state === "recording") {
      try { mr.stop(); } catch {}
    }
    mediaRecorderRef.current = null;

    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch {}
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      try { streamRef.current.getTracks().forEach((t) => t.stop()); } catch {}
      streamRef.current = null;
    }

    setIsRecording(false);
    setFeedback("Recording stopped.");
  };

  // apply new interval on the fly
  useEffect(() => {
    if (!isRecording) return;
    // restart the stop/start timer without touching the stream/recorder
    if (intervalRef.current) clearInterval(intervalRef.current);
    const mr = mediaRecorderRef.current;
    const periodMs = Math.max(1, intervalMin) * 60 * 1000;
    intervalRef.current = setInterval(() => {
      if (mr && mr.state === "recording") {
        try { mr.stop(); } catch {}
      }
    }, periodMs);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMin, isRecording]);

  // cleanup on unmount
  useEffect(() => () => { try { stopRecording(); } catch {} }, []);

  // -------- interventions --------
  const speak = (text) => {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1; u.pitch = 1; u.volume = 1;
    try { window.speechSynthesis.cancel(); } catch {}
    try { window.speechSynthesis.speak(u); } catch {}
  };

  const showRandomIntervention = () => {
    const options = ["breathing", "break", "dietVoice", "chatbot"];
    const pick = options[Math.floor(Math.random() * options.length)];
    setInterventionType(pick);
    setShowIntervention(true);

    if (pick === "dietVoice") {
      const lines = [
        "Try a rainbow bowl: blueberries, orange slices, spinach, carrots, avocado.",
        "Snack: apple with walnuts. Dinner: mixed greens, tomatoes, cucumber, olive oil.",
        "Keep it simple: banana, oats, and a leafy-green side salad.",
      ];
      const tip = lines[Math.floor(Math.random() * lines.length)];
      speak(`Quick stress-friendly tip: ${tip}`);
    }
    if (pick === "chatbot") {
      const evt = new CustomEvent("stressAlert", { detail: "stressed" });
      window.dispatchEvent(evt);
    }
  };

  // breathing coach
  const INHALE_MS = 30000, EXHALE_MS = 30000;
  const [breathingRunning, setBreathingRunning] = useState(false);
  const [breathingPhase, setBreathingPhase] = useState("idle");
  const [breathingProgress, setBreathingProgress] = useState(0);
  const breathingRunningRef = useRef(false);
  const breathingIntervalRef = useRef(null);
  const breathingPhaseTimerRef = useRef(null);

  const stopBreathing = useCallback(() => {
    breathingRunningRef.current = false;
    setBreathingRunning(false);
    setBreathingPhase("idle");
    setBreathingProgress(0);
    if (breathingIntervalRef.current) { clearInterval(breathingIntervalRef.current); breathingIntervalRef.current = null; }
    if (breathingPhaseTimerRef.current) { clearTimeout(breathingPhaseTimerRef.current); breathingPhaseTimerRef.current = null; }
  }, []);

  const runOneBreathingCycle = useCallback(() => {
    if (!breathingRunningRef.current) return;
    setBreathingPhase("inhale");
    setBreathingProgress(0);
    speak("Breathe in slowly.");
    const tick = 50;
    const startIn = Date.now();
    if (breathingIntervalRef.current) clearInterval(breathingIntervalRef.current);
    breathingIntervalRef.current = setInterval(() => {
      const p = Math.min(1, (Date.now() - startIn) / INHALE_MS);
      setBreathingProgress(p);
      if (p >= 1) clearInterval(breathingIntervalRef.current);
    }, tick);

    if (breathingPhaseTimerRef.current) clearTimeout(breathingPhaseTimerRef.current);
    breathingPhaseTimerRef.current = setTimeout(() => {
      if (!breathingRunningRef.current) return;
      setBreathingPhase("exhale");
      setBreathingProgress(1);
      speak("Now exhale gently.");
      const startEx = Date.now();
      if (breathingIntervalRef.current) clearInterval(breathingIntervalRef.current);
      breathingIntervalRef.current = setInterval(() => {
        const p = 1 - Math.min(1, (Date.now() - startEx) / EXHALE_MS);
        setBreathingProgress(p);
        if (p <= 0) clearInterval(breathingIntervalRef.current);
      }, tick);
      if (breathingPhaseTimerRef.current) clearTimeout(breathingPhaseTimerRef.current);
      breathingPhaseTimerRef.current = setTimeout(() => {
        if (breathingRunningRef.current) runOneBreathingCycle();
      }, EXHALE_MS);
    }, INHALE_MS);
  }, [EXHALE_MS, INHALE_MS]);

  const startBreathing = useCallback(() => {
    breathingRunningRef.current = true;
    setBreathingRunning(true);
    runOneBreathingCycle();
  }, [runOneBreathingCycle]);

  // break coach (pace up/down)
  const BREAK_TOTAL_SECONDS = 5 * 60;
  const [breakRunning, setBreakRunning] = useState(false);
  const [breakSecondsLeft, setBreakSecondsLeft] = useState(BREAK_TOTAL_SECONDS);
  const [paceUp, setPaceUp] = useState(true);
  const breakTickRef = useRef(null);

  const startBreak = useCallback(() => {
    setBreakSecondsLeft(BREAK_TOTAL_SECONDS);
    setPaceUp(true);
    setBreakRunning(true);
    speak("Let’s take a five minute break. Start walking. We will alternate pace up and pace down every thirty seconds.");
    if (breakTickRef.current) clearInterval(breakTickRef.current);
    breakTickRef.current = setInterval(() => {
      setBreakSecondsLeft((s) => {
        const next = Math.max(0, s - 1);
        const elapsed = BREAK_TOTAL_SECONDS - next;
        if (elapsed > 0 && elapsed % 30 === 0) {
          setPaceUp((prev) => {
            const nu = !prev;
            speak(nu ? "Pace up." : "Pace down.");
            return nu;
          });
        }
        if (next === 0) {
          clearInterval(breakTickRef.current);
          breakTickRef.current = null;
          setBreakRunning(false);
          speak("Break complete. Great job!");
        }
        return next;
      });
    }, 1000);
  }, []);

  const stopBreak = useCallback(() => {
    setBreakRunning(false);
    setBreakSecondsLeft(BREAK_TOTAL_SECONDS);
    setPaceUp(true);
    if (breakTickRef.current) {
      clearInterval(breakTickRef.current);
      breakTickRef.current = null;
    }
  }, []);

  const closeIntervention = () => {
    if (interventionType === "breathing") stopBreathing();
    if (interventionType === "break") stopBreak();
    setShowIntervention(false);
    setInterventionType(null);
  };

  // sizes
  const minSize = 80, maxSize = 220;
  const circleSize = Math.round(minSize + (maxSize - minSize) * breathingProgress);
  const mm = String(Math.floor(breakSecondsLeft / 60)).padStart(2, "0");
  const ss = String(breakSecondsLeft % 60).padStart(2, "0");

  return (
    <>
      <div className="dropdown-container">
        <label className="dropdown-label">Select Interval (in minutes):</label>
        <input
          type="number"
          min="1"
          value={intervalMin}
          onChange={(e) => setIntervalMin(Number(e.target.value) || 1)}
          className="select-dropdown"
        />
      </div>

      <div className="btn-container">
        <button className="success" onClick={startRecording} disabled={isRecording}>
          Start
        </button>
        <button className="danger" onClick={stopRecording} disabled={!isRecording}>
          Stop
        </button>
      </div>

      <canvas
        ref={canvasRef}
        id="pitchCanvas"
        width="600"
        height="100"
        style={{ border: "1px solid #007bff", marginTop: 20, backgroundColor: "#3A4A6B" }}
      />

      <div>
        {emotionLabel && <p className="emotion">{emotionLabel}</p>}
        {details && <p className="details">{details}</p>}
        {feedback && (
          <p className={`feedback ${!isRecording ? "feedback-stopped" : ""}`}>{feedback}</p>
        )}
      </div>

      <div className="filter-container">
        <label className="filter-label">Filter Results:</label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="select-dropdown"
        >
          <option value="All">All</option>
          <option value="Stressed">Stressed</option>
          <option value="Not Stressed">Not Stressed</option>
        </select>
      </div>

      <div className="emotion">
        <h2>Previous Results</h2>
        <div className="results-container">
          {filteredEmotions.length === 0 ? (
            <p style={{ opacity: 0.7 }}>No results yet. Start a recording to see your history here.</p>
          ) : (
            <ul>
              {filteredEmotions.map((entry, idx) => {
                const ts = parseTimestamp(entry.timestamp);
                const when = ts ? ts.toLocaleString() : "Unknown time";
                const label = (entry.emotion || "").toLowerCase() === "stressed" ? "Stressed" : "Not Stressed";
                return <li key={idx}>{when}: {label}</li>;
              })}
            </ul>
          )}
        </div>
      </div>

      {showIntervention && (
        <div className="bc-overlay show">
          <div className="bc-modal">
            <div className="bc-title">Quick Support</div>

            {interventionType === "breathing" && (
              <div className="bc-stage">
                <div className="bc-circle" style={{ width: circleSize, height: circleSize }} />
                <div className="bc-instructions">
                  {breathingPhase === "inhale" && "Breathe in…"}
                  {breathingPhase === "exhale" && "Breathe out…"}
                  {breathingPhase === "idle" && "Ready to begin guided breathing?"}
                </div>
                <div className="bc-actions">
                  {!breathingRunning ? (
                    <button className="bc-btn bc-start" onClick={startBreathing}>Start</button>
                  ) : (
                    <button className="bc-btn bc-stop" onClick={stopBreathing}>Stop</button>
                  )}
                  <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
                </div>
              </div>
            )}

            {interventionType === "break" && (
              <div className="bc-stage">
                <div className="break-timer">{mm}:{ss}</div>
                <div className="bc-instructions" style={{ textAlign: "center" }}>
                  {breakRunning
                    ? (paceUp ? "Pace up. Walk a bit faster." : "Pace down. Slow your steps.")
                    : "Press Start to begin a 5-minute walking break."}
                </div>
                <div className="bc-actions">
                  {!breakRunning ? (
                    <button className="bc-btn bc-start" onClick={startBreak}>Start</button>
                  ) : (
                    <button className="bc-btn bc-stop" onClick={stopBreak}>Stop</button>
                  )}
                  <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
                </div>
              </div>
            )}

            {interventionType === "dietVoice" && (
              <div className="bc-stage">
                <div className="bc-instructions" style={{ textAlign: "center" }}>
                  Playing a short voice tip on fruits and vegetables for stress…
                </div>
                <div className="bc-actions">
                  <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
                </div>
              </div>
            )}

            {interventionType === "chatbot" && (
              <div className="bc-stage">
                <div className="bc-instructions" style={{ textAlign: "center" }}>
                  Open the Stress Bot to chat right away.
                </div>
                <div className="bc-actions">
                  <button
                    className="bc-btn bc-start"
                    onClick={() => {
                      const evt = new CustomEvent("stressAlert", { detail: "stressed" });
                      window.dispatchEvent(evt);
                      closeIntervention();
                    }}
                  >
                    Open Stress Bot
                  </button>
                  <button className="bc-btn bc-close" onClick={closeIntervention}>Close</button>
                </div>
              </div>
            )}
          </div>

          <style>{`
            .bc-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.55); display: none; align-items: center; justify-content: center; z-index: 9999; }
            .bc-overlay.show { display: flex; }
            .bc-modal { width: 90%; max-width: 520px; background: #51558b; color: #fff; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
            .bc-title { text-align: center; font-weight: 700; font-size: 20px; margin-bottom: 12px; }
            .bc-stage { display: grid; place-items: center; gap: 12px; padding: 12px; }
            .bc-circle { border-radius: 50%; background: radial-gradient(circle at 30% 30%, #93a5ff, #6b73ff); transition: width 50ms linear, height 50ms linear; box-shadow: 0 6px 18px rgba(0,0,0,0.25) inset, 0 8px 22px rgba(0,0,0,0.2); }
            .bc-instructions { font-size: 18px; font-weight: 600; text-align:center; }
            .bc-actions { display: flex; gap: 10px; justify-content: center; margin-top: 12px; }
            .bc-btn { border: 0; padding: 10px 16px; border-radius: 999px; cursor: pointer; font-weight: 600; }
            .bc-start { background: #2ecc71; color: #08321b; }
            .bc-stop  { background: #ffcc66; color: #3a2b00; }
            .bc-close { background: #e25757; color: #fff; }
            .break-timer { font-size: 28px; font-weight: 700; letter-spacing: 1px; }
          `}</style>
        </div>
      )}
    </>
  );
};

export default Detector;
