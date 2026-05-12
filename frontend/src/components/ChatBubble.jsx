import React from 'react';

const ChatBubble = ({ message }) => {
  const isUser = message.role === 'user';
  
  const formatText = (text) => {
    if (!text) return <span className="empty-msg">...</span>;
    return text.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        {i < text.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-sender">{isUser ? 'You' : 'Kairos'}</div>
      <div className="message-bubble">
        <p>{formatText(message.text)}</p>
        

        
        {/* Masking alert */}
        {!isUser && message.masking_detected && (
          <div className="masking-tag">
            ⚠️ Masking detected
            {message.masking_details?.contradictions?.map((c, i) => (
              <div key={i} className="masking-detail">{c.detail}</div>
            ))}
          </div>
        )}
        
        {message.modality === 'audio' && (
          <div className="modality-tag">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 15C13.6569 15 15 13.6569 15 12V6C15 4.34315 13.6569 3 12 3C10.3431 3 9 4.34315 9 6V12C9 13.6569 10.3431 15 12 15Z" fill="currentColor"/>
              <path d="M19 10V12C19 15.866 15.866 19 12 19M5 10V12C5 15.866 8.13401 19 12 19M12 19V22M8 22H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Audio Message
          </div>
        )}
        {isUser && message.transcription && (
          <div className="transcription-text" style={{ fontSize: '0.85em', opacity: 0.8, marginTop: '8px', borderTop: '1px dashed rgba(255,255,255,0.2)', paddingTop: '8px' }}>
            <strong>Heard:</strong> {message.transcription}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatBubble;
