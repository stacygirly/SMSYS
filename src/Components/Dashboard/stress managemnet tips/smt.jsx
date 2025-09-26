/* import React, { useState } from 'react';
import { FaWalking, FaSpa, FaLeaf, FaAppleAlt, FaBed } from 'react-icons/fa';
import { IoIosArrowBack, IoIosArrowForward } from 'react-icons/io';
import './smt.css';
import exercise1Gif from '../../../assets/yoga.gif'; // Import your GIFs
import exercise2Gif from '../../../assets/break.gif';
import exercise3Gif from '../../../assets/food.gif';

const exercises = [
  {
    img: exercise1Gif,
    title: 'Exercise 1: Deep Breathing',
    description: 'Doing this will help to calm your mind and body.'
  },
  {
    img: exercise2Gif,
    title: 'Exercise 2: Take a Break',
    description: 'Take a break from work which causes stress.'
  },
  {
    img: exercise3Gif,
    title: 'Exercise 3: Do thing which makes you happy',
    description: 'Some people like to eat when there is stress, or some people like to travel Or some take a nap.'
  }
];

const StressManagementTips = () => {
  const [showPopup, setShowPopup] = useState(false);
  const [currentExercise, setCurrentExercise] = useState(0);

  const togglePopup = () => {
    setShowPopup(!showPopup);
  };

  const nextExercise = () => {
    setCurrentExercise((prev) => (prev + 1) % exercises.length);
  };

  const prevExercise = () => {
    setCurrentExercise((prev) => (prev - 1 + exercises.length) % exercises.length);
  };

  return (
    <div className="stress-management-tips-section">
      <h3>Stress Management Tips</h3>
      <ul>
        <li>
          <i><FaWalking /></i>
          <span>
            <div>Take regular breaks throughout the day.</div>
            <div>This helps to reduce stress and improve productivity. Short breaks can refresh your mind and body.</div>
          </span>
        </li>
        <li>
          <i><FaSpa /></i>
          <span>
            <div>Engage in physical activities such as walking or yoga.</div>
            <div>Physical activities can help manage stress. Exercise releases endorphins, which are natural mood lifters.</div>
          </span>
        </li>
        <li>
          <i><FaLeaf /></i>
          <span>
            <div>Practice mindfulness and relaxation techniques.</div>
            <div>Techniques like meditation can help calm your mind. Deep breathing exercises can reduce anxiety.</div>
          </span>
        </li>
        <li>
          <i><FaAppleAlt /></i>
          <span>
            <div>Maintain a healthy diet and stay hydrated.</div>
            <div>A good diet and hydration are essential for managing stress. Eating balanced meals can improve your overall mood.</div>
          </span>
        </li>
        <li>
          <i><FaBed /></i>
          <span>
            <div>Ensure you get enough sleep each night.</div>
            <div>Proper sleep is crucial for stress management. Aim for 7-9 hours of sleep each night to feel rested and alert.</div>
          </span>
        </li>
      </ul>
      <button className="btn smt-btn" onClick={togglePopup}>Try Relaxation Exercise</button>
      {showPopup && (
        <div className="popup">
          <div className="popup-inner">
            <h4>Relaxation Exercises</h4>
            <div className="exercise-card-container">
              <button className="arrow-btn" onClick={prevExercise}><IoIosArrowBack /></button>
              <div className="card">
                <img src={exercises[currentExercise].img} alt={exercises[currentExercise].title} className="exercise-gif" />
                <div>{exercises[currentExercise].title}</div>
                <p>{exercises[currentExercise].description}</p>
              </div>
              <button className="arrow-btn" onClick={nextExercise}><IoIosArrowForward /></button>
            </div>
            <button className="btn close-btn" onClick={togglePopup}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default StressManagementTips;
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FaWalking, FaSpa, FaLeaf, FaAppleAlt, FaBed } from 'react-icons/fa';
import { IoIosArrowBack, IoIosArrowForward } from 'react-icons/io';
import './smt.css';

import exercise1Gif from '../../../assets/yoga.gif';
import exercise2Gif from '../../../assets/break.gif';
import exercise3Gif from '../../../assets/food.gif';

const exercises = [
  {
    img: exercise1Gif,
    title: 'Exercise 1: Deep Breathing',
    description: 'Doing this will help to calm your mind and body.',
  },
  {
    img: exercise2Gif,
    title: 'Exercise 2: Take a Break',
    description: 'Take a break from work which causes stress.',
  },
  {
    img: exercise3Gif,
    title: 'Exercise 3: Do other things which makes you happy',
    description:
      'Some people like to eat when there is stress (remember to take some fruits and vegetables), or some people like to travel Or some take a nap.',
  },
];

const StressManagementTips = () => {
  const [showPopup, setShowPopup] = useState(false);
  const [currentExercise, setCurrentExercise] = useState(0);

  // ------------------ Exercise 1: Breathing coach ------------------
  const INHALE_MS = 30000;
  const EXHALE_MS = 30000;

  const [phase, setPhase] = useState('idle'); // inhale | exhale | idle
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0); // 0..1

  const runningRef = useRef(false);
  const intervalRef = useRef(null);
  const phaseTimerRef = useRef(null);

  // ------------------ Exercise 2: Break coach ------------------
  const BREAK_TOTAL_SECONDS = 5 * 60; // 5 minutes
  const [breakRunning, setBreakRunning] = useState(false);
  const [breakSecondsLeft, setBreakSecondsLeft] = useState(BREAK_TOTAL_SECONDS);
  const [paceUp, setPaceUp] = useState(true); // toggles every 30s

  const breakTickRef = useRef(null);

  // ------------------ Shared: chime + speech ------------------
  const chimeRef = useRef(null);
  const chimeSrc =
    'data:audio/mp3;base64,//uQZAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAACcQCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA';

  const speak = useCallback((text) => {
    if (!('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    u.pitch = 1;
    u.volume = 1;
    try { window.speechSynthesis.cancel(); } catch {}
    try { window.speechSynthesis.speak(u); } catch {}
  }, []);

  const playChime = useCallback(() => {
    try { chimeRef.current?.play().catch(() => {}); } catch {}
  }, []);

  // ------------------ Breathing helpers ------------------
  const clearBreathingTimers = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (phaseTimerRef.current) {
      clearTimeout(phaseTimerRef.current);
      phaseTimerRef.current = null;
    }
  }, []);

  const stopBreathing = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    setPhase('idle');
    setProgress(0);
    clearBreathingTimers();
  }, [clearBreathingTimers]);

  const runOneBreathingCycle = useCallback(() => {
    if (!runningRef.current) return;

    // INHALE
    playChime();
    setPhase('inhale');
    setProgress(0);
    const startIn = Date.now();
    const tick = 50;

    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      const p = Math.min(1, (Date.now() - startIn) / INHALE_MS);
      setProgress(p);
      if (p >= 1) clearInterval(intervalRef.current);
    }, tick);

    speak('Breathe in slowly.');

    if (phaseTimerRef.current) clearTimeout(phaseTimerRef.current);
    phaseTimerRef.current = setTimeout(() => {
      if (!runningRef.current) return;

      // EXHALE
      setPhase('exhale');
      setProgress(1);
      playChime();
      speak('Now exhale gently.');

      const startEx = Date.now();
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(() => {
        const p = 1 - Math.min(1, (Date.now() - startEx) / EXHALE_MS);
        setProgress(p);
        if (p <= 0) clearInterval(intervalRef.current);
      }, tick);

      if (phaseTimerRef.current) clearTimeout(phaseTimerRef.current);
      phaseTimerRef.current = setTimeout(() => {
        if (runningRef.current) runOneBreathingCycle(); // loop
      }, EXHALE_MS);
    }, INHALE_MS);
  }, [EXHALE_MS, INHALE_MS, playChime, speak]);

  const startBreathing = useCallback(() => {
    runningRef.current = true;
    setRunning(true);
    runOneBreathingCycle();
  }, [runOneBreathingCycle]);

  // ------------------ Break helpers ------------------
  const stopBreak = useCallback(() => {
    setBreakRunning(false);
    setBreakSecondsLeft(BREAK_TOTAL_SECONDS);
    setPaceUp(true);
    if (breakTickRef.current) {
      clearInterval(breakTickRef.current);
      breakTickRef.current = null;
    }
  }, []);

  const startBreak = useCallback(() => {
    // reset & go
    setBreakSecondsLeft(BREAK_TOTAL_SECONDS);
    setPaceUp(true);
    setBreakRunning(true);

    playChime();
    speak(
      'Let’s take a five minute break. Start walking around. We will alternate pace up and pace down every thirty seconds.'
    );

    if (breakTickRef.current) clearInterval(breakTickRef.current);
    breakTickRef.current = setInterval(() => {
      setBreakSecondsLeft((s) => {
        const next = Math.max(0, s - 1);

        // Every 30 seconds (including at 300→299 start), toggle cues
        const elapsed = BREAK_TOTAL_SECONDS - next; // seconds elapsed
        if (elapsed > 0 && elapsed % 30 === 0) {
          setPaceUp((prev) => {
            const nextPaceUp = !prev;
            playChime();
            speak(nextPaceUp ? 'Pace up.' : 'Pace down.');
            return nextPaceUp;
          });
        }

        if (next === 0) {
          clearInterval(breakTickRef.current);
          breakTickRef.current = null;
          setBreakRunning(false);
          playChime();
          speak('Break complete. Nice job!');
        }

        return next;
      });
    }, 1000);
  }, [playChime, speak]);

  // ------------------ Modal / slide cleanup ------------------
  useEffect(() => {
    if (!showPopup || currentExercise !== 0) stopBreathing();
    if (!showPopup || currentExercise !== 1) stopBreak();
  }, [showPopup, currentExercise, stopBreathing, stopBreak]);

  // circle size for breathing
  const minSize = 80;
  const maxSize = 220;
  const circleSize = Math.round(minSize + (maxSize - minSize) * progress);

  // mm:ss for break timer
  const mm = String(Math.floor(breakSecondsLeft / 60)).padStart(2, '0');
  const ss = String(breakSecondsLeft % 60).padStart(2, '0');

  const togglePopup = () => setShowPopup((s) => !s);
  const nextExercise = () => setCurrentExercise((prev) => (prev + 1) % exercises.length);
  const prevExercise = () =>
    setCurrentExercise((prev) => (prev - 1 + exercises.length) % exercises.length);

  return (
    <div className="stress-management-tips-section">
      <h3>Stress Management Tips</h3>
      <ul>
        <li>
          <i><FaWalking /></i>
          <span>
            <div>Take regular breaks throughout the day.</div>
            <div>This helps to reduce stress and improve productivity. Short breaks can refresh your mind and body.</div>
          </span>
        </li>
        <li>
          <i><FaSpa /></i>
          <span>
            <div>Engage in physical activities such as walking or yoga.</div>
            <div>Physical activities can help manage stress. Exercise releases endorphins, which are natural mood lifters.</div>
          </span>
        </li>
        <li>
          <i><FaLeaf /></i>
          <span>
            <div>Practice mindfulness and relaxation techniques.</div>
            <div>Techniques like meditation can help calm your mind. Deep breathing exercises can reduce anxiety.</div>
          </span>
        </li>
        <li>
          <i><FaAppleAlt /></i>
          <span>
            <div>Maintain a healthy diet and stay hydrated.</div>
            <div>A good diet and hydration are essential for managing stress. Eating balanced meals can improve your overall mood.</div>
          </span>
        </li>
        <li>
          <i><FaBed /></i>
          <span>
            <div>Ensure you get enough sleep each night.</div>
            <div>Proper sleep is crucial for stress management. Aim for 7-9 hours of sleep each night to feel rested and alert.</div>
          </span>
        </li>
      </ul>

      <button className="btn smt-btn" onClick={togglePopup}>Try Relaxation Exercise</button>

      {showPopup && (
        <div className="popup">
          <div className="popup-inner">
            <h4>Relaxation Exercises</h4>
            <audio ref={chimeRef} src={chimeSrc} preload="auto" />

            <div className="exercise-card-container">
              <button className="arrow-btn" onClick={prevExercise}><IoIosArrowBack /></button>

              <div className="card">
                <img
                  src={exercises[currentExercise].img}
                  alt={exercises[currentExercise].title}
                  className="exercise-gif"
                />
                <div>{exercises[currentExercise].title}</div>
                <p>{exercises[currentExercise].description}</p>

                {/* Exercise 1: Breathing */}
                {currentExercise === 0 && (
                  <div className="breathing-block">
                    <div
                      className="breathing-circle"
                      style={{ width: circleSize, height: circleSize }}
                    />
                    <div className="breathing-instructions">
                      {phase === 'inhale' && 'Breathe in…'}
                      {phase === 'exhale' && 'Breathe out…'}
                      {phase === 'idle' && 'Ready to begin guided breathing?'}
                    </div>
                    <div className="breathing-actions">
                      {!running ? (
                        <button className="btn start-btn" onClick={startBreathing}>Start</button>
                      ) : (
                        <button className="btn stop-btn" onClick={stopBreathing}>Stop</button>
                      )}
                    </div>
                  </div>
                )}

                {/* Exercise 2: 5-minute guided break */}
                {currentExercise === 1 && (
                  <div className="break-block">
                    <div className="break-timer">{mm}:{ss}</div>
                    <div className="break-instructions">
                      {breakRunning
                        ? (paceUp ? 'Pace up. Walk a bit faster.' : 'Pace down. Slow your steps.')
                        : 'Press Start to begin a 5-minute walking break.'}
                    </div>
                    <div className="break-actions">
                      {!breakRunning ? (
                        <button className="btn start-btn" onClick={startBreak}>Start</button>
                      ) : (
                        <button className="btn stop-btn" onClick={stopBreak}>Stop</button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <button className="arrow-btn" onClick={nextExercise}><IoIosArrowForward /></button>
            </div>

            <button className="btn close-btn" onClick={togglePopup}>Close</button>
          </div>

          {/* Scoped styles for the new blocks */}
          <style>{`
            .breathing-block { display:flex; flex-direction:column; align-items:center; gap:10px; margin-top:10px; }
            .breathing-circle {
              border-radius:50%;
              background: radial-gradient(circle at 30% 30%, #93a5ff, #6b73ff);
              transition: width 50ms linear, height 50ms linear;
              box-shadow: 0 6px 18px rgba(0,0,0,0.25) inset, 0 8px 22px rgba(0,0,0,0.2);
            }
            .breathing-instructions { font-weight:600; }
            .breathing-actions .start-btn { background:#2ecc71; color:#08321b; border:none; }
            .breathing-actions .stop-btn  { background:#ffcc66; color:#3a2b00; border:none; }

            .break-block { display:flex; flex-direction:column; align-items:center; gap:10px; margin-top:10px; }
            .break-timer { font-size:28px; font-weight:700; letter-spacing:1px; }
            .break-instructions { font-weight:600; text-align:center; }
            .break-actions .start-btn { background:#2ecc71; color:#08321b; border:none; }
            .break-actions .stop-btn  { background:#ffcc66; color:#3a2b00; border:none; }
          `}</style>
        </div>
      )}
    </div>
  );
};

export default StressManagementTips;
