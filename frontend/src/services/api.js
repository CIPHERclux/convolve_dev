/**
 * API Service — communicates with the FastAPI backend.
 */

const API_BASE = 'http://localhost:8000';

class ApiService {
  /**
   * Start a new session.
   * @param {string} userId
   * @returns {Promise<{session_id: string, user_id: string, message: string}>}
   */
  async startSession(userId = 'default_user') {
    const res = await fetch(`${API_BASE}/api/v1/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId }),
    });
    if (!res.ok) throw new Error(`Failed to start session: ${res.statusText}`);
    return res.json();
  }

  /**
   * End a session.
   * @param {string} sessionId
   */
  async endSession(sessionId) {
    const res = await fetch(`${API_BASE}/api/v1/session/${sessionId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error(`Failed to end session: ${res.statusText}`);
    return res.json();
  }

  /**
   * Send a text message.
   * @param {string} sessionId
   * @param {string} text
   * @returns {Promise<Object>} ChatResponse
   */
  async sendText(sessionId, text) {
    const res = await fetch(`${API_BASE}/api/v1/chat/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, text }),
    });
    if (!res.ok) throw new Error(`Chat failed: ${res.statusText}`);
    return res.json();
  }

  /**
   * Send an audio file with optional text.
   * @param {string} sessionId
   * @param {File} audioFile
   * @param {string} text
   * @returns {Promise<Object>} ChatResponse
   */
  async sendAudio(sessionId, audioFile, text = '') {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('text', text);
    formData.append('audio', audioFile);

    const res = await fetch(`${API_BASE}/api/v1/chat/audio`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(`Audio chat failed: ${res.statusText}`);
    return res.json();
  }

  /**
   * Health check.
   * @returns {Promise<Object>}
   */
  async health() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) return { status: 'error' };
      return res.json();
    } catch {
      return { status: 'error', qdrant_connected: false };
    }
  }

  /**
   * Get user profile.
   * @param {string} userId
   * @returns {Promise<Object>}
   */
  async getProfile(userId) {
    const res = await fetch(`${API_BASE}/api/v1/profile/${userId}`);
    if (!res.ok) throw new Error(`Profile fetch failed: ${res.statusText}`);
    return res.json();
  }
}

export const api = new ApiService();
export default api;
