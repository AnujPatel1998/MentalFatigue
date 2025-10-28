// FIXED Voice Analysis Handler
// Add this to your voice_handler.js or main JavaScript file

let mediaRecorder;
let audioChunks = [];
let audioBlob = null;

// Initialize voice recording
function initVoiceRecording() {
    const recordBtn = document.getElementById('record-btn');
    const stopBtn = document.getElementById('stop-btn');
    const analyzeBtn = document.getElementById('analyze-voice-btn');

    if (!recordBtn || !stopBtn || !analyzeBtn) {
        console.error('Voice buttons not found in DOM');
        return;
    }

    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    analyzeBtn.addEventListener('click', analyzeVoice);
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm'  // Changed from audio/wav
        });

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            console.log('Recording stopped. Blob size:', audioBlob.size);

            // Enable analyze button
            document.getElementById('analyze-voice-btn').disabled = false;

            // Show status
            updateStatus('Recording complete. Click Analyze to process.', 'success');
        };

        mediaRecorder.start();
        console.log('Recording started');

        // Update UI
        document.getElementById('record-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        updateStatus('Recording... Speak now!', 'info');

    } catch (error) {
        console.error('Error accessing microphone:', error);
        updateStatus('Error: Could not access microphone. Please grant permission.', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();

        // Stop all audio tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop());

        // Update UI
        document.getElementById('record-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
    }
}

async function analyzeVoice() {
    if (!audioBlob) {
        updateStatus('No recording found. Please record audio first.', 'error');
        return;
    }

    try {
        updateStatus('Analyzing voice...', 'info');

        // Create FormData
        const formData = new FormData();

        // CRITICAL FIX: Append audio file with correct field name
        formData.append('audio', audioBlob, 'recording.webm');

        // Add athlete ID
        const athleteId = sessionStorage.getItem('athlete_id') || 'anonymous';
        formData.append('athlete_id', athleteId);

        console.log('Sending audio blob:', audioBlob.size, 'bytes');

        // Send to backend
        const response = await fetch('http://localhost:5000/api/analyze/voice', {
            method: 'POST',
            body: formData  // Don't set Content-Type, browser will set it automatically
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            throw new Error(`Server returned ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        console.log('Analysis result:', result);

        if (result.success) {
            displayVoiceResults(result);
            updateStatus('Analysis complete!', 'success');
        } else {
            throw new Error(result.error || 'Analysis failed');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    }
}

function displayVoiceResults(result) {
    // Display fatigue score
    const scoreElement = document.getElementById('voice-fatigue-score');
    if (scoreElement) {
        scoreElement.textContent = result.fatigue_score.toFixed(1);
    }

    // Display fatigue level
    const levelElement = document.getElementById('voice-fatigue-level');
    if (levelElement) {
        levelElement.textContent = result.fatigue_level;
        levelElement.className = `fatigue-level ${result.fatigue_level.toLowerCase()}`;
    }

    // Display audio features
    const featuresElement = document.getElementById('audio-features');
    if (featuresElement && result.audio_features) {
        let featuresHTML = '<h3>Audio Features</h3><ul>';
        for (const [key, value] of Object.entries(result.audio_features)) {
            const displayName = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            featuresHTML += `<li><strong>${displayName}:</strong> ${value.toFixed(2)}</li>`;
        }
        featuresHTML += '</ul>';
        featuresElement.innerHTML = featuresHTML;
    }

    // Display recommendations
    const recsElement = document.getElementById('voice-recommendations');
    if (recsElement && result.recommendations) {
        let recsHTML = '<h3>Recommendations</h3><ul>';
        result.recommendations.forEach(rec => {
            recsHTML += `<li>${rec}</li>`;
        });
        recsHTML += '</ul>';
        recsElement.innerHTML = recsHTML;
    }

    // Show results section
    const resultsSection = document.getElementById('voice-results');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
}

function updateStatus(message, type) {
    const statusElement = document.getElementById('voice-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
    }
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVoiceRecording);
} else {
    initVoiceRecording();
}
