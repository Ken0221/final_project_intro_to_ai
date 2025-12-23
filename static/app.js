/**
 * Bowling Scorer - Frontend Application
 * ======================================
 * Handles WebSocket communication, UI updates, and game flow.
 */

// ==========================================
// Global State
// ==========================================

const state = {
    socket: null,
    currentScreen: 'settings',
    previousScreen: null,  // Track previous screen for back navigation
    players: [],
    gameState: null,
    lastThrowInfo: null,
    arduinoConnected: false
};

// ==========================================
// Initialization
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializeEventListeners();
    updateFramesValue();
});

function initializeSocket() {
    // Connect to SocketIO server
    state.socket = io({
        transports: ['websocket', 'polling']
    });

    // Connection events
    state.socket.on('connect', () => {
        console.log('Connected to server');
        showToast('Connected to server', 'success');
    });

    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        showToast('Disconnected from server', 'error');
    });

    // Video frame updates
    state.socket.on('video_frame', (data) => {
        updateVideoFeed(data.frame);
    });

    // Player crop updates
    state.socket.on('player_crop', (data) => {
        updatePlayerCrop(data.crop, data.player_name);
    });

    // Game state updates
    state.socket.on('game_state', (data) => {
        updateGameState(data);
    });

    // Arduino status
    state.socket.on('arduino_status', (data) => {
        updateArduinoStatus(data.connected);
    });

    // Tracking status
    state.socket.on('tracking_status', (data) => {
        updateTrackingStatus(data.target, data.found);
    });

    // Countdown updates
    state.socket.on('countdown', (data) => {
        showCountdown(data.seconds);
    });

    // Score update notifications
    state.socket.on('score_update', (data) => {
        handleScoreUpdate(data);
    });

    // Registration result
    state.socket.on('registration_result', (data) => {
        handleRegistrationResult(data);
    });

    // Game over
    state.socket.on('game_over', (data) => {
        showResults(data);
    });

    // Error handling
    state.socket.on('error', (data) => {
        showToast(data.message, 'error');
    });
}

function initializeEventListeners() {
    // Settings screen
    document.getElementById('numFrames').addEventListener('input', updateFramesValue);
    document.getElementById('startSetupBtn').addEventListener('click', startSetup);
    document.getElementById('settingsBtn').addEventListener('click', toggleSettings);

    // Registration screen
    document.getElementById('regPlayer1Btn').addEventListener('click', () => registerPlayer(1));
    document.getElementById('regPlayer2Btn').addEventListener('click', () => registerPlayer(2));
    document.getElementById('startGameBtn').addEventListener('click', startGame);

    // Game screen
    document.getElementById('correctScoreBtn').addEventListener('click', showCorrectionModal);

    // Modal
    document.getElementById('cancelCorrectionBtn').addEventListener('click', hideCorrectionModal);
    document.getElementById('confirmCorrectionBtn').addEventListener('click', confirmCorrection);

    // Results screen
    document.getElementById('playAgainBtn').addEventListener('click', playAgain);
}

// ==========================================
// Screen Navigation
// ==========================================

function showScreen(screenName, savePrevious = false) {
    // Save previous screen if requested
    if (savePrevious && state.currentScreen !== screenName) {
        state.previousScreen = state.currentScreen;
    }

    // Hide all screens
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });

    // Show target screen
    const targetScreen = document.getElementById(screenName + 'Screen');
    if (targetScreen) {
        targetScreen.classList.add('active');
        state.currentScreen = screenName;
    }

    // Update settings button text
    updateSettingsButton();
}

function toggleSettings() {
    if (state.currentScreen === 'settings') {
        // If we're in settings, go back to previous screen
        if (state.previousScreen) {
            showScreen(state.previousScreen);
            state.previousScreen = null;
        }
    } else {
        // Save current screen and go to settings
        showScreen('settings', true);
    }
}

function updateSettingsButton() {
    const btn = document.getElementById('settingsBtn');
    if (state.currentScreen === 'settings' && state.previousScreen) {
        btn.textContent = '← Back';
        btn.title = 'Back to game';
    } else {
        btn.textContent = '⚙️';
        btn.title = 'Settings';
    }
}

// ==========================================
// Settings Screen
// ==========================================

function updateFramesValue() {
    const value = document.getElementById('numFrames').value;
    document.getElementById('framesValue').textContent = value;
}

function startSetup() {
    const comPort = document.getElementById('comPort').value;
    const cameraSource = document.getElementById('cameraSource').value;
    const numFrames = parseInt(document.getElementById('numFrames').value);
    const numBottles = parseInt(document.getElementById('numBottles').value) || 10;
    const yoloModel = document.getElementById('yoloModel').value;
    const reidBackbone = document.getElementById('reidBackbone').value;

    // Send settings to server
    state.socket.emit('set_settings', {
        com_port: comPort,
        camera_source: cameraSource,
        num_frames: numFrames,
        num_bottles: numBottles,
        yolo_model: yoloModel,
        reid_backbone: reidBackbone
    });

    showScreen('registration');
    showToast(`Loading models: ${yoloModel}, ${reidBackbone}...`, 'info');
}

// ==========================================
// Registration Screen
// ==========================================

function registerPlayer(playerNum) {
    const nameInput = document.getElementById(`player${playerNum}Name`);
    const name = nameInput.value.trim();

    if (!name) {
        showToast('Please enter a player name', 'warning');
        return;
    }

    // Send registration request
    state.socket.emit('register_player', {
        player_num: playerNum,
        name: name
    });

    showToast(`Capturing ${name}...`, 'info');
}

function handleRegistrationResult(data) {
    if (data.success) {
        const card = document.getElementById(`player${data.player_num}Card`);
        const status = document.getElementById(`player${data.player_num}Status`);
        const btn = document.getElementById(`regPlayer${data.player_num}Btn`);

        card.classList.add('registered');
        status.textContent = `✓ ${data.name} registered`;
        btn.disabled = true;
        btn.textContent = '✓ Registered';

        state.players[data.player_num - 1] = data.name;

        showToast(`${data.name} registered successfully!`, 'success');

        // Enable start game if both players registered
        if (state.players.filter(p => p).length === 2) {
            document.getElementById('startGameBtn').disabled = false;
        }
    } else {
        showToast(data.message || 'Registration failed', 'error');
    }
}

function startGame() {
    state.socket.emit('start_game');
    showScreen('game');
    showToast('Game started! Good luck! 🎳', 'success');
}

// ==========================================
// Game Screen
// ==========================================

function updateVideoFeed(frameData) {
    const img = state.currentScreen === 'registration'
        ? document.getElementById('regVideoFeed')
        : document.getElementById('gameVideoFeed');

    if (img) {
        img.src = 'data:image/jpeg;base64,' + frameData;
    }
}

function updatePlayerCrop(cropData, playerName) {
    const cropImg = document.getElementById('playerCropFeed');
    const cropName = document.getElementById('cropPlayerName');

    if (cropImg && cropData) {
        cropImg.src = 'data:image/jpeg;base64,' + cropData;
    }
    if (cropName) {
        cropName.textContent = playerName || 'Current Player';
    }
}

function updateGameState(data) {
    state.gameState = data;

    // Update game info bar
    document.getElementById('currentFrame').textContent =
        `${data.current_frame}/${data.num_frames}`;
    document.getElementById('currentThrow').textContent =
        `${data.current_throw}/2`;
    document.getElementById('currentPlayerName').textContent =
        data.current_player_name || '---';

    // Update scoreboard
    updateScoreboard(data.players);

    // Handle game over
    if (data.state === 'game_over') {
        showResults(data);
    }
}

function updateScoreboard(players) {
    const tbody = document.getElementById('scoreboardBody');
    if (!tbody || !players) return;

    tbody.innerHTML = '';

    players.forEach((player, playerIdx) => {
        const row = document.createElement('tr');
        row.className = 'player-row';
        if (state.gameState && state.gameState.current_player_index === playerIdx) {
            row.classList.add('active');
        }

        // Player name
        const nameCell = document.createElement('td');
        nameCell.className = 'player-name';
        nameCell.textContent = player.name;
        row.appendChild(nameCell);

        // Frame cells
        player.frames.forEach((frame, frameIdx) => {
            const cell = document.createElement('td');
            const isTenth = frameIdx === 9;

            cell.innerHTML = `
                <div class="frame-cell">
                    <div class="frame-throws">
                        <div class="throw-box ${frame.is_strike ? 'strike' : ''}">${frame.throw1 || ''}</div>
                        <div class="throw-box ${frame.is_spare ? 'spare' : ''}">${frame.throw2 || ''}</div>
                        ${isTenth ? `<div class="throw-box">${frame.throw3 || ''}</div>` : ''}
                    </div>
                    <div class="frame-score">${frame.score !== null ? frame.score : ''}</div>
                </div>
            `;
            row.appendChild(cell);
        });

        // Total score
        const totalCell = document.createElement('td');
        totalCell.className = 'total-col';
        totalCell.textContent = player.total_score;
        row.appendChild(totalCell);

        tbody.appendChild(row);
    });
}

function updateTrackingStatus(target, found) {
    const overlay = document.getElementById('trackingOverlay');
    const targetSpan = document.getElementById('trackingTarget');

    if (targetSpan) {
        const targetNames = {
            'person': '👤 Player',
            'tennis_ball': '🎾 Ball',
            'bottle': '🎳 Bottles'
        };
        targetSpan.textContent = `Tracking: ${targetNames[target] || target}`;
    }
}

function updateArduinoStatus(connected) {
    state.arduinoConnected = connected;
    const statusEl = document.getElementById('arduinoStatus');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('span:last-child');

    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Arduino: Connected';
    } else {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Arduino: Disconnected';
    }
}

function showCountdown(seconds) {
    const overlay = document.getElementById('countdownOverlay');
    const number = document.getElementById('countdownNumber');

    if (seconds > 0) {
        overlay.style.display = 'flex';
        number.textContent = seconds;
    } else {
        overlay.style.display = 'none';
    }
}

function handleScoreUpdate(data) {
    state.lastThrowInfo = data;

    let message = `${data.player}: `;
    if (data.is_strike) {
        message += 'STRIKE! 🎯';
    } else if (data.is_spare) {
        message += 'SPARE! ✨';
    } else {
        message += `${data.pins_knocked} pins`;
    }

    showToast(message, 'success');

    // Update standing pins display
    const standingPins = 10 - data.pins_knocked;
    document.getElementById('standingPins').textContent =
        data.is_strike || data.frame_complete ? '10' : standingPins;
}

// ==========================================
// Score Correction
// ==========================================

function showCorrectionModal() {
    if (!state.lastThrowInfo) {
        showToast('No throw to correct', 'warning');
        return;
    }

    const info = state.lastThrowInfo;
    document.getElementById('correctionInfo').textContent =
        `Last throw: ${info.player}, Frame ${info.frame}, Throw ${info.throw}`;
    document.getElementById('correctedPins').value = info.pins_knocked;

    document.getElementById('correctionModal').classList.add('active');
}

function hideCorrectionModal() {
    document.getElementById('correctionModal').classList.remove('active');
}

function confirmCorrection() {
    const newPins = parseInt(document.getElementById('correctedPins').value);

    if (isNaN(newPins) || newPins < 0 || newPins > 10) {
        showToast('Please enter a valid number (0-10)', 'warning');
        return;
    }

    state.socket.emit('correct_score', {
        pins_knocked: newPins
    });

    hideCorrectionModal();
    showToast('Score corrected', 'success');
}

// ==========================================
// Results Screen
// ==========================================

function showResults(data) {
    showScreen('results');

    // Update winner announcement
    const announcement = document.getElementById('winnerAnnouncement');
    if (data.winner === 'Tie') {
        announcement.textContent = "It's a Tie! 🤝";
    } else {
        announcement.textContent = `Winner: ${data.winner}! 🏆`;
    }

    // Update final scores
    const scoresContainer = document.getElementById('finalScores');
    scoresContainer.innerHTML = '';

    data.players.forEach(player => {
        const card = document.createElement('div');
        card.className = 'final-score-card';
        if (player.name === data.winner) {
            card.classList.add('winner');
        }

        card.innerHTML = `
            <div class="name">${player.name}</div>
            <div class="score">${player.total_score}</div>
        `;

        scoresContainer.appendChild(card);
    });
}

function playAgain() {
    state.socket.emit('reset_game');

    // Reset UI state
    state.players = [];
    state.lastThrowInfo = null;

    // Reset registration UI
    [1, 2].forEach(num => {
        const card = document.getElementById(`player${num}Card`);
        const status = document.getElementById(`player${num}Status`);
        const btn = document.getElementById(`regPlayer${num}Btn`);
        const input = document.getElementById(`player${num}Name`);

        card.classList.remove('registered');
        status.textContent = '';
        btn.disabled = false;
        btn.textContent = '📷 Capture & Register';
        input.value = '';
    });

    document.getElementById('startGameBtn').disabled = true;

    showScreen('registration');
    showToast('Ready for a new game!', 'success');
}

// ==========================================
// Toast Notifications
// ==========================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ==========================================
// Utility Functions
// ==========================================

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
