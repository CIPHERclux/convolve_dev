import React, { useState, useEffect, useRef } from 'react';
import api from './services/api';
import ChatBubble from './components/ChatBubble';

function App() {
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('offline'); // online, offline, error
  
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Check health on mount
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const checkHealth = async () => {
    try {
      const res = await api.health();
      setStatus(res.status === 'ok' ? 'online' : 'offline');
    } catch {
      setStatus('offline');
    }
  };

  const handleStartSession = async () => {
    try {
      setIsLoading(true);
      const res = await api.startSession();
      setSession(res);
      setMessages([{ role: 'assistant', text: res.message, modality: 'text' }]);
    } catch (error) {
      console.error("Failed to start session:", error);
      alert("Backend is offline or unreachable. Please ensure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendText = async (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || !session || isLoading) return;

    const userText = inputText.trim();
    setInputText('');
    setMessages(prev => [...prev, { role: 'user', text: userText, modality: 'text' }]);
    setIsLoading(true);

    try {
      const res = await api.sendText(session.session_id, userText);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: res.response_text,
        modality: 'text',
        emotion_tag: res.emotion_tag,
        masking_detected: res.masking_detected,
        masking_details: res.masking_details,
        biomarkers: res.biomarkers,
      }]);
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages(prev => [...prev, { role: 'assistant', text: "I'm having trouble connecting right now. Please try again.", modality: 'text' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  };

  const startRecording = async () => {
    if (!session || isLoading) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
        
        // Stop all tracks to release mic
        stream.getTracks().forEach(track => track.stop());
        
        await sendAudioMessage(audioFile);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please ensure permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioMessage = async (audioFile) => {
    setMessages(prev => [...prev, { role: 'user', text: '[Voice Message Sent]', modality: 'audio' }]);
    setIsLoading(true);

    try {
      const res = await api.sendAudio(session.session_id, audioFile);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: res.response_text,
        modality: 'text',
        emotion_tag: res.emotion_tag,
        masking_detected: res.masking_detected,
        masking_details: res.masking_details,
        biomarkers: res.biomarkers,
      }]);
    } catch (error) {
      console.error("Failed to send audio:", error);
      setMessages(prev => [...prev, { role: 'assistant', text: "I couldn't process that audio. Please try again or type a message.", modality: 'text' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="chat-container">
        {/* Header */}
        <header className="header">
          <div className="brand">
            <div className="brand-icon">K</div>
            <div className="brand-name">Kairos</div>
          </div>
          <div className="system-status">
            <div className={`status-dot ${status}`}></div>
            <span>System {status === 'online' ? 'Active' : 'Offline'}</span>
          </div>
        </header>

        {/* Main Content Area */}
        {!session ? (
          <div className="start-container">
            <div className="empty-state">
              <h2>Welcome to Kairos</h2>
              <p>A multimodal mental health support companion that listens to what you say and how you say it.</p>
            </div>
            <button 
              className="btn-primary" 
              onClick={handleStartSession}
              disabled={isLoading || status !== 'online'}
            >
              {isLoading ? 'Starting...' : 'Begin Session'}
            </button>
          </div>
        ) : (
          <>
            {/* Mental State Visualizer */}
            {(() => {
              const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant' && m.emotion_tag);
              if (!lastAssistantMsg) return null;
              
              const isMasking = lastAssistantMsg.masking_detected;
              const emotion = lastAssistantMsg.emotion_tag || 'neutral';
              
              return (
                <div className="state-visualizer">
                  <div className="state-item">
                    <span className="state-label">Detected Emotion</span>
                    <span className={`emotion-badge emotion--${emotion}`}>{emotion.toUpperCase()}</span>
                  </div>
                  <div className="state-item">
                    <span className="state-label">Masking Status</span>
                    {isMasking ? (
                      <span className="masking-badge active">⚠️ High Probability</span>
                    ) : (
                      <span className="masking-badge safe">✓ Authentic</span>
                    )}
                  </div>
                </div>
              );
            })()}

            <div className="messages-area">
              {messages.map((msg, index) => (
                <ChatBubble key={index} message={msg} />
              ))}
              {isLoading && (
                <div className="message-wrapper assistant">
                  <div className="message-sender">Kairos</div>
                  <div className="message-bubble typing-indicator">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="input-container">
              <div className="input-box">
                <div className="textarea-wrapper">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type a message..."
                    rows={1}
                    disabled={isLoading || isRecording}
                  />
                </div>
                
                {/* Record Button */}
                <button 
                  className={`action-btn ${isRecording ? 'recording' : ''}`}
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onTouchStart={startRecording}
                  onTouchEnd={stopRecording}
                  disabled={isLoading}
                  title="Hold to record"
                >
                  {isRecording ? (
                    <svg viewBox="0 0 24 24">
                      <rect x="7" y="7" width="10" height="10" rx="2" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24">
                      <path d="M12 14C13.6569 14 15 12.6569 15 11V5C15 3.34315 13.6569 2 12 2C10.3431 2 9 3.34315 9 5V11C9 12.6569 10.3431 14 12 14Z" />
                      <path d="M19 10V11C19 14.866 15.866 18 12 18M5 10V11C5 14.866 8.13401 18 12 18M12 18V22M8 22H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </button>
                
                {/* Send Button */}
                <button 
                  className="action-btn send" 
                  onClick={handleSendText}
                  disabled={isLoading || !inputText.trim() || isRecording}
                >
                  <svg viewBox="0 0 24 24">
                    <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
