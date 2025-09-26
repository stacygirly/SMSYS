/* import React, { useState, useEffect } from 'react';
import { FaRegMessage } from "react-icons/fa6";
import { IoMdSend } from "react-icons/io";
import { BsRobot } from "react-icons/bs";
import { AiOutlineLoading3Quarters } from "react-icons/ai"; // Importing loading icon
import axios from 'axios';
import './chatbot.css';

const Chatbot = ({ username, isAuthenticated }) => {
    const [showChat, setShowChat] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState("");
    const [inputHeight, setInputHeight] = useState("40px");
    const [isLoading, setIsLoading] = useState(false);

    const toggleChat = () => {
        setShowChat(!showChat);
    };

    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    };

    const handleSend = async () => {
        if (inputValue.trim() !== "") {
            const newMessages = [...messages, { role: 'user', content: inputValue }];
            setMessages(newMessages);
            setInputValue("");
            setInputHeight("40px");
            setIsLoading(true);

            try {
                const response = await axios.post('http://localhost:5000/chat', { message: inputValue, username }, { withCredentials: true });
                const botMessage = response.data.message;
                setMessages([...newMessages, { role: 'assistant', content: botMessage }]);
            } catch (error) {
                console.error('Error communicating with ChatGPT API:', error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            setInputHeight("80px");
            e.preventDefault();
            handleSend();
        }
    };

    const getInitials = (name) => {
        if (!name) return '';
        return name.split(' ').map((part) => part[0]).join('').toUpperCase();
    };

    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                const response = await axios.get('http://localhost:5000/chats', { withCredentials: true });
                if (response.status === 200) {
                    setMessages(response.data.flatMap(chat => chat.messages));
                }
            } catch (error) {
                console.error('Error fetching chat history:', error);
            }
        };

        if (isAuthenticated) {
            fetchChatHistory();
        }

        const handleStressAlert = (event) => {
            const { detail } = event;
            if (detail === 'stressed') {
                const alertMessage = `
                    Hi ${username}, it seems you are experiencing stress. Here are some tips to help you manage:
                    1. Take regular breaks throughout the day to refresh your mind and body.
                    2. Engage in physical activities such as walking or yoga to release endorphins and lift your mood.
                    3. Practice mindfulness and relaxation techniques like meditation and deep breathing exercises to calm your mind.
                    4. Maintain a healthy diet and stay hydrated to support your overall well-being.
                    5. Ensure you get enough sleep each night to feel rested and alert.
                    6. Please contact your family doctor for further guidance.
                    If you need further assistance, feel free to ask me!
                `;
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: alertMessage }]);
                setShowChat(true); // Automatically open chat on stress alert
            }
        };

        window.addEventListener('stressAlert', handleStressAlert);

        return () => {
            window.removeEventListener('stressAlert', handleStressAlert);
        };
    }, [username, isAuthenticated]);

    return (
        <div>
            {isAuthenticated && (
                <div className='message' onClick={toggleChat}>
                    <FaRegMessage className='message-icon' />
                </div>
            )}
            {showChat && (
                <div className='chat-ui'>
                    <div className='chat-header'>
                        Stress Bot
                        <button className='close-chat' onClick={toggleChat}>✖</button>
                    </div>
                    <div className='chat-body'>
                        {messages.map((msg, index) => (
                            <div key={index} className={`chat-message ${msg.role}`}>
                                {msg.role === 'user' ? (
                                    <div className="avatar">{getInitials(username)}</div>
                                ) : (
                                    <div className="avatar bot-avatar">
                                        <BsRobot className="bot-icon" />
                                    </div>
                                )}
                                <div className="message-content">
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className='chat-message assistant'>
                                <div className="avatar bot-avatar">
                                    <BsRobot className="bot-icon" />
                                </div>
                                <div className="message-content">
                                    <AiOutlineLoading3Quarters className="loading-icon" />
                                </div>
                            </div>
                        )}
                    </div>
                    <div className='chat-footer'>
                        <textarea 
                            placeholder='Type a message...'
                            value={inputValue}
                            onChange={handleInputChange}
                            onKeyPress={handleKeyPress}
                            style={{ height: inputHeight }}
                        />
                        <IoMdSend className='send-icon' onClick={handleSend} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default Chatbot; */
/* import React, { useState, useEffect, useRef } from 'react';
import { FaRegMessage } from "react-icons/fa6";
import { IoMdSend } from "react-icons/io";
import { BsRobot } from "react-icons/bs";
import { AiOutlineLoading3Quarters } from "react-icons/ai";
import axios from 'axios';
import './chatbot.css';

const Chatbot = ({ username, isAuthenticated }) => {
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [inputHeight, setInputHeight] = useState("40px");
  const [isLoading, setIsLoading] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true); // allow muting TTS if desired

  const bodyRef = useRef(null);

  const toggleChat = () => setShowChat((s) => !s);

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    // simple textarea growth up to 120px
    const base = 40;
    const lines = e.target.value.split('\n').length;
    const newH = Math.min(base + (lines - 1) * 20, 120);
    setInputHeight(`${newH}px`);
  };

  const speak = (text) => {
    if (!ttsEnabled) return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // stop anything pending
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    } else {
      console.warn('Text-to-speech not supported in this browser.');
    }
  };

  const handleSend = async () => {
    const content = inputValue.trim();
    if (!content) return;

    const newMessages = [...messages, { role: 'user', content }];
    setMessages(newMessages);
    setInputValue("");
    setInputHeight("40px");
    setIsLoading(true);

    try {
      const response = await axios.post(
        'http://localhost:5000/chat',
        { message: content, username },
        { withCredentials: true }
      );
      const botMessage = response?.data?.message ?? "(no response)";
      setMessages([...newMessages, { role: 'assistant', content: botMessage }]);
    } catch (error) {
      console.error('Error communicating with ChatGPT API:', error);
      const msg = error?.response?.status === 401
        ? "Please sign in to chat."
        : "Network error. Please try again.";
      setMessages([...newMessages, { role: 'assistant', content: msg }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // prevent newline
      handleSend();
      return;
    }
    // allow Shift+Enter for newline
  };

  const getInitials = (name) => {
    if (!name) return '';
    return name.split(' ').map((part) => part[0]).join('').toUpperCase();
  };

  // auto-scroll to latest
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages, isLoading, showChat]);

  useEffect(() => {
    const fetchChatHistory = async () => {
      try {
        const response = await axios.get('http://localhost:5000/chats', { withCredentials: true });
        if (response.status === 200) {
          setMessages(response.data.flatMap(chat => chat.messages || []));
        }
      } catch (error) {
        if (error?.response?.status !== 401) {
          console.error('Error fetching chat history:', error);
        }
      }
    };

    if (isAuthenticated) {
      fetchChatHistory();
    }

    const handleStressAlert = (event) => {
      const { detail } = event;
      if (detail === 'stressed') {
        const alertMessage = `
Hi ${username || 'there'}, it seems you are experiencing stress. Here are some tips to help you manage:
1. Take regular breaks throughout the day to refresh your mind and body.
2. Engage in physical activities such as walking or yoga to release endorphins and lift your mood.
3. Practice mindfulness and relaxation techniques like meditation and deep breathing exercises to calm your mind.
4. Maintain a healthy diet and stay hydrated to support your overall well-being.
5. Ensure you get enough sleep each night to feel rested and alert.
6. Please contact your family doctor for further guidance.
If you need further assistance, feel free to ask me!
        `.trim();

        // push message
        setMessages(prev => [...prev, { role: 'assistant', content: alertMessage }]);

        // speak aloud
        speak(alertMessage);

        // open chat if closed
        setShowChat(true);
      }
    };

    window.addEventListener('stressAlert', handleStressAlert);
    return () => {
      window.removeEventListener('stressAlert', handleStressAlert);
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    };
  }, [username, isAuthenticated, ttsEnabled]);

  return (
    <div>
      {isAuthenticated && (
        <div className='message' onClick={toggleChat}>
          <FaRegMessage className='message-icon' />
        </div>
      )}

      {showChat && (
        <div className='chat-ui'>
          <div className='chat-header'>
            Stress Bot
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label style={{ fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={ttsEnabled}
                  onChange={(e) => setTtsEnabled(e.target.checked)}
                  style={{ marginRight: 6 }}
                />
                Voice
              </label>
              <button className='close-chat' onClick={toggleChat}>✖</button>
            </div>
          </div>

          <div className='chat-body' ref={bodyRef}>
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                {msg.role === 'user' ? (
                  <div className="avatar">{getInitials(username)}</div>
                ) : (
                  <div className="avatar bot-avatar">
                    <BsRobot className="bot-icon" />
                  </div>
                )}
                <div className="message-content">
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className='chat-message assistant'>
                <div className="avatar bot-avatar">
                  <BsRobot className="bot-icon" />
                </div>
                <div className="message-content">
                  <AiOutlineLoading3Quarters className="loading-icon" />
                </div>
              </div>
            )}
          </div>

          <div className='chat-footer'>
            <textarea
              placeholder='Type a message...'
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              style={{ height: inputHeight }}
            />
            <IoMdSend className='send-icon' onClick={handleSend} />
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot; */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FaRegMessage } from "react-icons/fa6";
import { IoMdSend, IoMdMic } from "react-icons/io";
import { BsRobot } from "react-icons/bs";
import { AiOutlineLoading3Quarters } from "react-icons/ai";
import axios from 'axios';
import './chatbot.css';

const Chatbot = ({ username, isAuthenticated }) => {
    const [showChat, setShowChat] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState("");
    const [inputHeight, setInputHeight] = useState("40px");
    const [isLoading, setIsLoading] = useState(false);
    const recognitionRef = useRef(null);
    const audioRef = useRef(null);

    const toggleChat = () => setShowChat(!showChat);
    const handleInputChange = (e) => setInputValue(e.target.value);

    // Speak function wrapped in useCallback
    const speak = useCallback((text) => {
        if (audioRef.current) {
            audioRef.current.play().catch(err => console.warn("Chime play error:", err));
        }
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";
        window.speechSynthesis.speak(utterance);
    }, []);

    const handleSend = async () => {
        if (inputValue.trim() !== "") {
            const newMessages = [...messages, { role: 'user', content: inputValue }];
            setMessages(newMessages);
            setInputValue("");
            setInputHeight("40px");
            setIsLoading(true);

            try {
                const response = await axios.post('https://persuasive.research.cs.dal.ca/smsys/chat', { message: inputValue, username }, { withCredentials: true });
                const botMessage = response.data.message;
                setMessages([...newMessages, { role: 'assistant', content: botMessage }]);
                speak(botMessage);
            } catch (error) {
                console.error('Error communicating with ChatGPT API:', error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            setInputHeight("80px");
            e.preventDefault();
            handleSend();
        }
    };

    const startListening = () => {
        if (!('webkitSpeechRecognition' in window)) {
            alert("Speech recognition not supported in this browser.");
            return;
        }

        if (!recognitionRef.current) {
            const recognition = new window.webkitSpeechRecognition();
            recognition.lang = "en-US";
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                setInputValue(transcript);
                handleSend();
            };

            recognition.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
            };

            recognitionRef.current = recognition;
        }

        if (window.confirm("Do you want to ask me now?")) {
            recognitionRef.current.start();
        }
    };

    const getInitials = (name) => (!name ? '' : name.split(' ').map((part) => part[0]).join('').toUpperCase());

    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                const response = await axios.get('https://persuasive.research.cs.dal.ca/smsys/chats', { withCredentials: true });
                if (response.status === 200) {
                    setMessages(response.data.flatMap(chat => chat.messages));
                }
            } catch (error) {
                console.error('Error fetching chat history:', error);
            }
        };

        if (isAuthenticated) {
            fetchChatHistory();
        }

        const handleStressAlert = (event) => {
            const { detail } = event;
            if (detail === 'stressed') {
                const alertMessage = `
                    Hi ${username}, it seems you are experiencing stress. Here are some tips to help you manage:
                    1. Take regular breaks throughout the day to refresh your mind and body.
                    2. Engage in physical activities such as walking or yoga to release endorphins and lift your mood.
                    3. Practice mindfulness and relaxation techniques like meditation and deep breathing exercises to calm your mind.
                    4. Maintain a healthy diet and stay hydrated to support your overall well-being.
                    5. Ensure you get enough sleep each night to feel rested and alert.
                    6. Please contact your family doctor for further guidance.
                    If you need further assistance, feel free to ask me!
                `;
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: alertMessage }]);
                speak(alertMessage);
                setShowChat(true);
            }
        };

        window.addEventListener('stressAlert', handleStressAlert);
        return () => window.removeEventListener('stressAlert', handleStressAlert);
    }, [username, isAuthenticated, speak]); // speak added here

    return (
        <div>
            {/* Chime audio element */}
            <audio ref={audioRef} src="/chime.mp3" preload="auto"></audio>

            {isAuthenticated && (
                <div className='message' onClick={toggleChat}>
                    <FaRegMessage className='message-icon' />
                </div>
            )}
            {showChat && (
                <div className='chat-ui'>
                    <div className='chat-header'>
                        Stress Bot
                        <button className='close-chat' onClick={toggleChat}>✖</button>
                    </div>
                    <div className='chat-body'>
                        {messages.map((msg, index) => (
                            <div key={index} className={`chat-message ${msg.role}`}>
                                {msg.role === 'user' ? (
                                    <div className="avatar">{getInitials(username)}</div>
                                ) : (
                                    <div className="avatar bot-avatar">
                                        <BsRobot className="bot-icon" />
                                    </div>
                                )}
                                <div className="message-content">
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className='chat-message assistant'>
                                <div className="avatar bot-avatar">
                                    <BsRobot className="bot-icon" />
                                </div>
                                <div className="message-content">
                                    <AiOutlineLoading3Quarters className="loading-icon" />
                                </div>
                            </div>
                        )}
                    </div>
                    <div className='chat-footer'>
                        <textarea
                            placeholder='Type a message...'
                            value={inputValue}
                            onChange={handleInputChange}
                            onKeyPress={handleKeyPress}
                            style={{ height: inputHeight }}
                        />
                        <IoMdMic className='mic-icon' onClick={startListening} title="Speak to the bot" />
                        <IoMdSend className='send-icon' onClick={handleSend} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default Chatbot;
