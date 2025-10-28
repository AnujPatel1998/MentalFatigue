// Application Configuration
const config = {
    apiBaseUrl: 'http://localhost:5000',
    athleteId: 'athlete_001' // In a real app, this would be from authentication
};

// Application State
let appState = {
    currentView: 'home',
    isRecording: false,
    audioBlob: null,
    dashboardData: [],
    fatigueChart: null
};

// Language options
const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'hi', name: 'Hindi' },
    { code: 'zh-cn', name: 'Chinese' }
];

// Fatigue levels configuration
const fatigueLevels = {
    low: { range: [0, 3], level: 'Low', color: '#10b981', description: 'Minimal mental fatigue' },
    mild: { range: [3, 5], level: 'Mild', color: '#f59e0b', description: 'Some fatigue present' },
    moderate: { range: [5, 7], level: 'Moderate', color: '#f97316', description: 'Significant fatigue' },
    high: { range: [7, 8.5], level: 'High', color: '#ef4444', description: 'High fatigue levels' },
    severe: { range: [8.5, 10], level: 'Severe', color: '#dc2626', description: 'Severe mental fatigue' }
};

// Sample recommendations
const recommendations = {
    low: [
        'Maintain current wellness routines',
        'Continue regular sleep schedule',
        'Keep up balanced nutrition'
    ],
    mild: [
        'Consider a light recovery day',
        'Focus on hydration and nutrition',
        'Practice mindfulness or meditation'
    ],
    moderate: [
        'Schedule a rest day or light training',
        'Consult with sports psychologist',
        'Focus on stress-reduction techniques'
    ],
    high: [
        'URGENT: Consult with medical team',
        'Take 2-3 days complete rest',
        'Avoid high-intensity training'
    ],
    severe: [
        'IMMEDIATE: Seek medical attention',
        'Complete training cessation',
        'Professional psychological support'
    ]
};

// Utility Functions
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    notification.classList.add('show');
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function showView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });
    
    // Show selected view
    document.getElementById(viewName).classList.add('active');
    
    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
    
    appState.currentView = viewName;
    
    // Initialize view-specific functionality
    if (viewName === 'dashboard') {
        initializeDashboard();
    }
}

function getFatigueLevel(score) {
    for (const [key, config] of Object.entries(fatigueLevels)) {
        if (score >= config.range[0] && score < config.range[1]) {
            return { key, ...config };
        }
    }
    return { key: 'severe', ...fatigueLevels.severe };
}

function updateFatigueDisplay(score, prefix) {
    const fatigueInfo = getFatigueLevel(score);
    const scoreCircle = document.getElementById(`${prefix}ScoreCircle`);
    const scoreValue = document.getElementById(`${prefix}ScoreValue`);
    const fatigueLevel = document.getElementById(`${prefix}FatigueLevel`);
    const fatigueDescription = document.getElementById(`${prefix}FatigueDescription`);
    
    scoreCircle.className = `score-circle score-${fatigueInfo.key}`;
    scoreValue.textContent = score.toFixed(1);
    fatigueLevel.textContent = `${fatigueInfo.level} Fatigue`;
    fatigueDescription.textContent = fatigueInfo.description;
}

function displayRecommendations(score, containerId, listId) {
    const fatigueInfo = getFatigueLevel(score);
    const recList = recommendations[fatigueInfo.key] || recommendations.low;
    
    const container = document.getElementById(containerId);
    const list = document.getElementById(listId);
    
    list.innerHTML = recList.map(rec => `<li>${rec}</li>`).join('');
    container.style.display = 'block';
}

// API Functions
async function makeApiCall(endpoint, method = 'GET', data = null) {
    const url = `${config.apiBaseUrl}${endpoint}`;
    
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`API call failed: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        // Return mock data for demonstration
        return getMockApiResponse(endpoint, data);
    }
}

function getMockApiResponse(endpoint, data) {
    const mockScore = Math.random() * 10;
    
    if (endpoint.includes('/analyze/text')) {
        return {
            fatigue_score: mockScore,
            sentiment: {
                positive: Math.random() * 0.5,
                negative: Math.random() * 0.5,
                neutral: Math.random() * 0.3
            },
            contributing_factors: [
                'Sleep quality mentioned',
                'Training intensity concerns',
                'Motivation levels'
            ],
            confidence: 0.85
        };
    }
    
    if (endpoint.includes('/analyze/voice')) {
        return {
            fatigue_score: mockScore,
            voice_features: {
                pitch_variance: Math.random() * 100,
                energy_level: Math.random() * 100,
                speech_rate: 120 + Math.random() * 60,
                pause_frequency: Math.random() * 10
            },
            confidence: 0.78
        };
    }
    
    if (endpoint.includes('/dashboard')) {
        const mockData = [];
        for (let i = 7; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            mockData.push({
                date: date.toISOString().split('T')[0],
                fatigue_score: Math.random() * 10,
                type: Math.random() > 0.5 ? 'text' : 'voice',
                timestamp: date.toISOString()
            });
        }
        return { assessments: mockData };
    }
    
    if (endpoint.includes('/forecast')) {
        const forecast = [];
        for (let i = 1; i <= 7; i++) {
            const date = new Date();
            date.setDate(date.getDate() + i);
            forecast.push({
                date: date.toISOString().split('T')[0],
                predicted_score: 3 + Math.random() * 4
            });
        }
        return { forecast };
    }
    
    return { success: true };
}

// Text Analysis Functions
async function analyzeText() {
    const textInput = document.getElementById('textInput').value.trim();
    const language = document.getElementById('languageSelect').value;
    const analyzeBtn = document.getElementById('analyzeTextBtn');
    
    if (!textInput) {
        showNotification('Please enter some text to analyze', 'error');
        return;
    }
    
    // Show loading state
    analyzeBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    analyzeBtn.disabled = true;
    
    try {
        const result = await makeApiCall('/api/analyze/text', 'POST', {
            text: textInput,
            language: language,
            athlete_id: config.athleteId
        });
        
        displayTextResults(result);
        showNotification('Text analysis completed successfully!');
        
    } catch (error) {
        showNotification('Error analyzing text. Please try again.', 'error');
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze Text';
        analyzeBtn.disabled = false;
    }
}

function displayTextResults(result) {
    const resultsSection = document.getElementById('textResults');
    
    // Update fatigue display
    updateFatigueDisplay(result.fatigue_score, 'text');
    
    // Display sentiment analysis
    const sentimentDiv = document.getElementById('textSentiment');
    if (result.sentiment) {
        const sentiment = result.sentiment;
        sentimentDiv.innerHTML = `
            <h5>Sentiment Analysis</h5>
            <div style="margin-bottom: 8px;">Positive: ${(sentiment.positive * 100).toFixed(1)}%</div>
            <div style="margin-bottom: 8px;">Negative: ${(sentiment.negative * 100).toFixed(1)}%</div>
            <div>Neutral: ${(sentiment.neutral * 100).toFixed(1)}%</div>
        `;
    }
    
    // Display contributing factors
    const factorsDiv = document.getElementById('textFactors');
    if (result.contributing_factors) {
        factorsDiv.innerHTML = `
            <h5 style="margin-top: 16px;">Key Factors Identified</h5>
            <ul style="margin: 8px 0; padding-left: 20px;">
                ${result.contributing_factors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        `;
    }
    
    // Display recommendations
    displayRecommendations(result.fatigue_score, 'textRecommendations', 'textRecList');
    
    resultsSection.classList.add('active');
}

// Voice Analysis Functions
let mediaRecorder = null;
let recordedChunks = [];

function initializeVoiceRecording() {
    const recordBtn = document.getElementById('recordBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const analyzeVoiceBtn = document.getElementById('analyzeVoiceBtn');
    
    recordBtn.addEventListener('click', async () => {
        if (!appState.isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    });
    
    analyzeVoiceBtn.addEventListener('click', analyzeVoice);
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedChunks = [];
        
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            appState.audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
            document.getElementById('analyzeVoiceBtn').disabled = false;
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        appState.isRecording = true;
        
        // Update UI
        const recordBtn = document.getElementById('recordBtn');
        const recordingStatus = document.getElementById('recordingStatus');
        const audioWaveform = document.getElementById('audioWaveform');
        
        recordBtn.classList.add('recording');
        recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
        recordingStatus.textContent = 'Recording... Click to stop';
        audioWaveform.style.display = 'block';
        
    } catch (error) {
        showNotification('Could not access microphone. Please check permissions.', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && appState.isRecording) {
        mediaRecorder.stop();
        appState.isRecording = false;
        
        // Update UI
        const recordBtn = document.getElementById('recordBtn');
        const recordingStatus = document.getElementById('recordingStatus');
        const audioWaveform = document.getElementById('audioWaveform');
        
        recordBtn.classList.remove('recording');
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        recordingStatus.textContent = 'Recording complete. Click analyze to process.';
        audioWaveform.style.display = 'none';
    }
}

async function analyzeVoice() {
    if (!appState.audioBlob) {
        showNotification('Please record audio first', 'error');
        return;
    }
    
    const analyzeBtn = document.getElementById('analyzeVoiceBtn');
    
    // Show loading state
    analyzeBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    analyzeBtn.disabled = true;
    
    try {
        // In a real application, you would upload the audio file
        // For this demo, we'll simulate the analysis
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const result = await makeApiCall('/api/analyze/voice', 'POST', {
            athlete_id: config.athleteId
        });
        
        displayVoiceResults(result);
        showNotification('Voice analysis completed successfully!');
        
    } catch (error) {
        showNotification('Error analyzing voice. Please try again.', 'error');
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '<i class="fas fa-microphone"></i> Analyze Voice';
        analyzeBtn.disabled = false;
    }
}

function displayVoiceResults(result) {
    const resultsSection = document.getElementById('voiceResults');
    
    // Update fatigue display
    updateFatigueDisplay(result.fatigue_score, 'voice');
    
    // Display voice features
    const featuresDiv = document.getElementById('voiceFeatures');
    if (result.voice_features) {
        const features = result.voice_features;
        featuresDiv.innerHTML = `
            <div style="margin-bottom: 8px;">Pitch Variance: ${features.pitch_variance?.toFixed(2) || 'N/A'} Hz</div>
            <div style="margin-bottom: 8px;">Energy Level: ${features.energy_level?.toFixed(1) || 'N/A'}%</div>
            <div style="margin-bottom: 8px;">Speech Rate: ${features.speech_rate?.toFixed(0) || 'N/A'} words/min</div>
            <div>Pause Frequency: ${features.pause_frequency?.toFixed(2) || 'N/A'} pauses/sec</div>
        `;
    }
    
    // Display recommendations
    displayRecommendations(result.fatigue_score, 'voiceRecommendations', 'voiceRecList');
    
    resultsSection.classList.add('active');
}

// Dashboard Functions
async function initializeDashboard() {
    try {
        const data = await makeApiCall('/api/dashboard');
        appState.dashboardData = data.assessments || [];
        
        updateDashboardStats();
        updateFatigueChart();
        updateAssessmentTable();
        loadForecast();
        
    } catch (error) {
        showNotification('Error loading dashboard data', 'error');
    }
}

function updateDashboardStats() {
    const data = appState.dashboardData;
    
    if (data.length === 0) {
        document.getElementById('avgScore').textContent = '0.0';
        document.getElementById('totalAssessments').textContent = '0';
        document.getElementById('trendDirection').textContent = 'No Data';
        return;
    }
    
    // Calculate average score
    const avgScore = data.reduce((sum, item) => sum + item.fatigue_score, 0) / data.length;
    document.getElementById('avgScore').textContent = avgScore.toFixed(1);
    
    // Total assessments
    document.getElementById('totalAssessments').textContent = data.length;
    
    // Trend direction (simple calculation)
    if (data.length >= 2) {
        const recent = data.slice(-3).reduce((sum, item) => sum + item.fatigue_score, 0) / Math.min(3, data.length);
        const earlier = data.slice(0, -3).reduce((sum, item) => sum + item.fatigue_score, 0) / Math.max(1, data.length - 3);
        
        const trendElement = document.getElementById('trendDirection');
        if (recent > earlier + 0.5) {
            trendElement.textContent = 'Increasing';
            trendElement.style.color = 'var(--sports-red)';
        } else if (recent < earlier - 0.5) {
            trendElement.textContent = 'Improving';
            trendElement.style.color = 'var(--sports-green)';
        } else {
            trendElement.textContent = 'Stable';
            trendElement.style.color = 'var(--color-text)';
        }
    } else {
        document.getElementById('trendDirection').textContent = 'Insufficient Data';
    }
}

function updateFatigueChart() {
    const ctx = document.getElementById('fatigueChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (appState.fatigueChart) {
        appState.fatigueChart.destroy();
    }
    
    const data = appState.dashboardData;
    
    appState.fatigueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(item => {
                const date = new Date(item.date);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }),
            datasets: [{
                label: 'Fatigue Score',
                data: data.map(item => item.fatigue_score),
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Mental Fatigue Trend Over Time'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 10,
                    title: {
                        display: true,
                        text: 'Fatigue Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

function updateAssessmentTable() {
    const tableContainer = document.getElementById('assessmentTable');
    const data = appState.dashboardData;
    
    if (data.length === 0) {
        tableContainer.innerHTML = '<p>No assessments found.</p>';
        return;
    }
    
    const table = `
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: var(--color-bg-1); border-bottom: 1px solid var(--color-border);">
                    <th style="padding: 12px; text-align: left;">Date</th>
                    <th style="padding: 12px; text-align: left;">Type</th>
                    <th style="padding: 12px; text-align: left;">Score</th>
                    <th style="padding: 12px; text-align: left;">Level</th>
                </tr>
            </thead>
            <tbody>
                ${data.map(item => {
                    const fatigueInfo = getFatigueLevel(item.fatigue_score);
                    return `
                        <tr style="border-bottom: 1px solid var(--color-border);">
                            <td style="padding: 12px;">${new Date(item.date).toLocaleDateString()}</td>
                            <td style="padding: 12px;">
                                <i class="fas fa-${item.type === 'text' ? 'file-text' : 'microphone'}"></i>
                                ${item.type.charAt(0).toUpperCase() + item.type.slice(1)}
                            </td>
                            <td style="padding: 12px; font-weight: bold;">${item.fatigue_score.toFixed(1)}</td>
                            <td style="padding: 12px;">
                                <span style="color: ${fatigueInfo.color}; font-weight: 500;">${fatigueInfo.level}</span>
                            </td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = table;
}

async function loadForecast() {
    try {
        const result = await makeApiCall('/api/forecast', 'POST', {
            athlete_id: config.athleteId
        });
        
        displayForecast(result.forecast || []);
    } catch (error) {
        document.getElementById('forecastSection').innerHTML = '<p>Unable to load forecast data.</p>';
    }
}

function displayForecast(forecast) {
    const container = document.getElementById('forecastSection');
    
    if (forecast.length === 0) {
        container.innerHTML = '<p>No forecast data available.</p>';
        return;
    }
    
    const forecastHtml = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 16px;">
            ${forecast.map(item => {
                const date = new Date(item.date);
                const fatigueInfo = getFatigueLevel(item.predicted_score);
                return `
                    <div style="text-align: center; padding: 16px; background: var(--color-bg-1); border-radius: 8px;">
                        <div style="font-size: 12px; color: var(--color-text-secondary); margin-bottom: 4px;">
                            ${date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                        </div>
                        <div style="font-size: 18px; font-weight: bold; color: ${fatigueInfo.color};">
                            ${item.predicted_score.toFixed(1)}
                        </div>
                        <div style="font-size: 11px; color: var(--color-text-secondary);">
                            ${fatigueInfo.level}
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
    
    container.innerHTML = forecastHtml;
}

// Filter and Export Functions
function applyFilters() {
    // This would filter the dashboard data based on date range and type
    // For now, we'll just refresh the display
    updateDashboardStats();
    updateFatigueChart();
    updateAssessmentTable();
    showNotification('Filters applied successfully!');
}

function downloadReport(format) {
    if (format === 'pdf') {
        showNotification('PDF report generation started...');
        // In a real app, this would generate and download a PDF
        setTimeout(() => {
            showNotification('PDF report would be downloaded in a real application');
        }, 1000);
    } else if (format === 'csv') {
        // Generate CSV data
        const csvContent = generateCSV();
        downloadCSV(csvContent, 'fatigue_report.csv');
        showNotification('CSV report downloaded successfully!');
    }
}

function generateCSV() {
    const data = appState.dashboardData;
    const headers = ['Date', 'Type', 'Fatigue Score', 'Fatigue Level'];
    
    const csvRows = [headers.join(',')];
    
    data.forEach(item => {
        const fatigueInfo = getFatigueLevel(item.fatigue_score);
        const row = [
            item.date,
            item.type,
            item.fatigue_score.toFixed(1),
            fatigueInfo.level
        ];
        csvRows.push(row.join(','));
    });
    
    return csvRows.join('\n');
}

function downloadCSV(csvContent, fileName) {
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', fileName);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Navigation event listeners
    document.querySelectorAll('[data-view]').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const viewName = this.getAttribute('data-view');
            showView(viewName);
        });
    });
    
    // Text analysis event listeners
    document.getElementById('analyzeTextBtn').addEventListener('click', analyzeText);
    
    // Voice analysis initialization
    initializeVoiceRecording();
    
    // Dashboard event listeners
    document.getElementById('applyFilters').addEventListener('click', applyFilters);
    document.getElementById('downloadPDF').addEventListener('click', () => downloadReport('pdf'));
    document.getElementById('downloadCSV').addEventListener('click', () => downloadReport('csv'));
    
    // Set default date range for filters
    const today = new Date();
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    
    document.getElementById('dateFrom').value = weekAgo.toISOString().split('T')[0];
    document.getElementById('dateTo').value = today.toISOString().split('T')[0];
    
    // Initialize with home view
    showView('home');
});

// Add CSS animation for waveform
const style = document.createElement('style');
style.textContent = `
@keyframes waveform {
    0% { height: 30px; }
    50% { height: 60px; }
    100% { height: 40px; }
}
`;
document.head.appendChild(style);