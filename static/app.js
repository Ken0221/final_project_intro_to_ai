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
    arduinoConnected: false,
    numFrames: 0, // Track number of frames to rebuild header only when needed
    numBottles: 10, // Track number of bottles for standing pins display

    // Track captured angles for each player
    registrationState: {
        1: { front: false, back: false, left: false, right: false },
        2: { front: false, back: false, left: false, right: false }
    }
};

// ==========================================
// Initialization
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializeEventListeners();
    updateFramesValue();
    // Fetch registered players for the dropdowns
    refreshRegisteredPlayers();
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

    // Wrong player warning
    state.socket.on('wrong_player_warning', (data) => {
        const overlay = document.getElementById('wrongPlayerOverlay');
        if (overlay) {
            overlay.style.display = data.show ? 'flex' : 'none';
        }
    });

    // Score update notifications
    state.socket.on('score_update', (data) => {
        handleScoreUpdate(data);
    });

    // Registration result
    // Registration result (Finalize)
    state.socket.on('registration_result', (data) => {
        handleRegistrationResult(data);
    });

    // Capture result (Single Angle)
    state.socket.on('capture_result', (data) => {
        handleCaptureResult(data);
    });

    // Game over
    state.socket.on('game_over', (data) => {
        showResults(data);
    });

    // Error handling
    state.socket.on('error', (data) => {
        showToast(data.message, 'error');
    });

    // Temporal smoothing status update
    state.socket.on('temporal_smoothing_status', (data) => {
        updateTemporalSmoothingButton(data.enabled);
    });
}

function initializeEventListeners() {
    // Settings screen
    document.getElementById('numFrames').addEventListener('input', updateFramesValue);
    document.getElementById('startSetupBtn').addEventListener('click', startSetup);
    document.getElementById('settingsBtn').addEventListener('click', toggleSettings);

    // Registration screen
    // Registration screen (buttons use inline onclick handlers)
    // document.getElementById('regPlayer1Btn').addEventListener('click', () => registerPlayer(1));
    // document.getElementById('regPlayer2Btn').addEventListener('click', () => registerPlayer(2));
    document.getElementById('startGameBtn').addEventListener('click', startGame);

    // Game screen
    document.getElementById('correctScoreBtn').addEventListener('click', showCorrectionModal);

    // Modal
    document.getElementById('cancelCorrectionBtn').addEventListener('click', hideCorrectionModal);
    document.getElementById('confirmCorrectionBtn').addEventListener('click', confirmCorrection);

    // Results screen
    document.getElementById('playAgainBtn').addEventListener('click', playAgain);

    // Manage Saves Modal
    document.getElementById('manageSavesBtn').addEventListener('click', openManageSavesModal);
    document.getElementById('closeManageSavesBtn').addEventListener('click', closeManageSavesModal);
    document.getElementById('selectAllSaves').addEventListener('change', toggleSelectAllSaves);
    document.getElementById('deleteSelectedSavesBtn').addEventListener('click', deleteSelectedSaves);

    // Close modal if clicking outside content
    window.addEventListener('click', (event) => {
        const modal = document.getElementById('manageSavesModal');
        if (event.target === modal) {
            closeManageSavesModal();
        }
    });

    // Camera Control (Manual)
    setupCameraControl('btnPanLeft', 'pan', -1);
    setupCameraControl('btnPanRight', 'pan', 1);
    setupCameraControl('btnTiltUp', 'tilt', 1); // Up button -> Angle Decrease (Look Up) = direction 1 (positive logic for button, app handles sign)
    setupCameraControl('btnTiltDown', 'tilt', -1); // Down button -> Angle Increase (Look Down)
}

function setupCameraControl(btnId, axis, direction) {
    const btn = document.getElementById(btnId);
    if (!btn) return;

    // Helper to send command
    const sendCmd = (stop = false) => {
        state.socket.emit('manual_motor_control', {
            axis: axis,
            direction: direction,
            stop: stop
        });
    };

    if (axis === 'pan') {
        // Pan: Hold to move (start on down, stop on up)
        const startMove = (e) => {
            e.preventDefault(); // Prevent text selection/scrolling
            sendCmd(false);
        };
        const stopMove = (e) => {
            e.preventDefault();
            sendCmd(true);
        };

        btn.addEventListener('mousedown', startMove);
        btn.addEventListener('touchstart', startMove);

        // Stop on release or leaving button
        btn.addEventListener('mouseup', stopMove);
        btn.addEventListener('mouseleave', stopMove);
        btn.addEventListener('touchend', stopMove);
    } else {
        // Tilt: Hold to move (start on down, stop on up) -> Repeat steps
        let intervalId = null;

        const startMove = (e) => {
            e.preventDefault();
            if (intervalId) return;

            sendCmd(false); // Send immediate first step
            intervalId = setInterval(() => sendCmd(false), 100); // Repeat every 100ms
        };

        const stopMove = (e) => {
            e.preventDefault();
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
            // For tilt, "Stop" command isn't strictly needed as it's positional step, 
            // but harmless to stop interval. We don't need to send a "Stop" command to backend 
            // because backend doesn't have "Tilt Speed", only "Tilt Angle".
            // But let's verify if `sendCmd(true)` does anything for tilt.
            // Backend `handle_manual_motor_control`: 
            // `stop = data.get('stop')` 
            // For tilt, `stop` is currently ignored in backend logic I wrote?
            // Actually, looking at `app.py`: `stop` is used for Pan. For Tilt, it just steps.
            // So we just stop the interval.
        };

        btn.addEventListener('mousedown', startMove);
        btn.addEventListener('touchstart', startMove);

        btn.addEventListener('mouseup', stopMove);
        btn.addEventListener('mouseleave', stopMove);
        btn.addEventListener('touchend', stopMove);
    }
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

    // Refresh registered players after setup (gives backend time to initialize)
    setTimeout(refreshRegisteredPlayers, 1000);
}

// ==========================================
// Registered Player Selection
// ==========================================

// Fetch registered players from the server
async function refreshRegisteredPlayers() {
    try {
        const response = await fetch('/api/registered_players');
        const data = await response.json();
        const players = data.players || [];

        // Populate both player dropdowns
        populateRegisteredPlayerSelect(1, players);
        populateRegisteredPlayerSelect(2, players);

        if (players.length > 0) {
            console.log(`Loaded ${players.length} registered players:`, players);
        }
    } catch (error) {
        console.error('Failed to fetch registered players:', error);
    }
}

// Populate the dropdown for a specific player slot
function populateRegisteredPlayerSelect(playerNum, players) {
    const select = document.getElementById(`player${playerNum}RegisteredSelect`);
    if (!select) return;

    // Clear existing options except the default
    select.innerHTML = '<option value="">-- Select --</option>';

    // Add player options
    players.forEach(playerName => {
        const option = document.createElement('option');
        option.value = playerName;
        option.textContent = playerName;
        select.appendChild(option);
    });
}

// Handle selecting a registered player from dropdown
function onRegisteredPlayerSelect(playerNum) {
    const select = document.getElementById(`player${playerNum}RegisteredSelect`);
    const nameInput = document.getElementById(`player${playerNum}Name`);
    const selectedName = select.value;

    if (!selectedName) return;

    // Update the name input field
    nameInput.value = selectedName;

    // Send request to select this player directly from database
    state.socket.emit('select_registered_player', {
        player_num: playerNum,
        name: selectedName
    });

    showToast(`Loading ${selectedName} from database...`, 'info');
}

// ==========================================
// Registration Screen
// ==========================================

// Capture a single angle for a player
function captureAngle(playerNum, angle) {
    const name = document.getElementById(`player${playerNum}Name`).value.trim();
    if (!name) {
        showToast('Please enter a name first', 'warning');
        return;
    }

    // Disable button while capturing
    const btn = document.getElementById(`p${playerNum}${capitalize(angle)}Btn`);
    if (btn) btn.disabled = true;

    state.socket.emit('register_capture', {
        player_num: playerNum,
        angle: angle
    });

    showToast(`Capturing ${angle}...`, 'info');
}

// Finalize registration after capturing all angles
function finalizeRegistration(playerNum) {
    const name = document.getElementById(`player${playerNum}Name`).value.trim();
    if (!name) {
        showToast('Please enter a name', 'warning');
        return;
    }

    state.socket.emit('finalize_registration', {
        player_num: playerNum,
        name: name
    });

    showToast(`Finalizing registration for ${name}...`, 'info');
}

// Handle capture result from server
function handleCaptureResult(data) {
    const playerNum = data.player_num;
    const angle = data.angle;
    const btnId = `p${playerNum}${capitalize(angle)}Btn`;
    const btn = document.getElementById(btnId);
    const indicator = document.getElementById(`p${playerNum}Ind${capitalize(angle)}`);

    if (data.success) {
        state.registrationState[playerNum][angle] = true;

        // Update button appearance
        if (btn) {
            btn.classList.add('captured');
            btn.innerHTML = `✓ ${capitalize(angle)}`;
        }

        // Update indicator
        if (indicator) {
            indicator.classList.add('active');
        }

        showToast(`Captured ${angle} for Player ${playerNum}`, 'success');

        // Check if all angles captured (enable register button when at least 1 captured)
        checkReadyToRegister(playerNum);

    } else {
        showToast(data.message || 'Capture failed', 'error');
        // Re-enable button
        if (btn) btn.disabled = false;
    }
}

// Check if player is ready to register (all 4 angles captured)
function checkReadyToRegister(playerNum) {
    const angles = state.registrationState[playerNum];
    const count = Object.values(angles).filter(v => v).length;
    const regBtn = document.getElementById(`regPlayer${playerNum}Btn`);

    // Enable register button when all 4 angles captured
    if (count >= 4 && regBtn) {
        regBtn.disabled = false;
    }
}

// Capitalize first letter
function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
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
        cropName.textContent = `Current Player: ${playerName || 'Unknown'}`;
    }
}

function updateGameState(data) {
    state.gameState = data;

    // Check if we need to update the scoreboard header
    if (state.numFrames !== data.num_frames) {
        state.numFrames = data.num_frames;
        renderScoreboardHeader(data.num_frames);
    }

    // Track num_bottles for standing pins display
    if (data.num_bottles) {
        state.numBottles = data.num_bottles;
    }

    // Update game info bar
    document.getElementById('currentFrame').textContent =
        `${data.current_frame}/${data.num_frames}`;
    document.getElementById('currentThrow').textContent =
        `${data.current_throw}/2`;
    document.getElementById('currentPlayerName').textContent =
        data.current_player_name || '---';

    // Update scoreboard
    updateScoreboard(data.players, data.num_frames);

    // Handle game over
    if (data.state === 'game_over') {
        showResults(data);
    }
}

function renderScoreboardHeader(numFrames) {
    const headerRow = document.getElementById('scoreboardHeaderRow');
    if (!headerRow) return;

    // Clear existing frame headers (keep Player and Total)
    // We expect: [Player, ...frames..., Total]
    // Player is index 0, Total is last.

    // Remove all children except first and last? No, easier to rebuild.
    headerRow.innerHTML = '';

    // Player Column
    const playerTh = document.createElement('th');
    playerTh.className = 'player-name-col';
    playerTh.textContent = 'Player';
    headerRow.appendChild(playerTh);

    // Frame Columns
    for (let i = 1; i <= numFrames; i++) {
        const th = document.createElement('th');
        th.textContent = i;
        if (i === numFrames) {
            th.className = 'frame-10'; // Keep style for last frame if special
        }
        headerRow.appendChild(th);
    }

    // Total Column
    const totalTh = document.createElement('th');
    totalTh.className = 'total-col';
    totalTh.textContent = 'Total';
    headerRow.appendChild(totalTh);
}

function updateScoreboard(players, numFrames) {
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
        // Render cells up to numFrames. Player might have fewer frames if data is weird, 
        // but normally player.frames matches numFrames (initialized in backend).
        for (let i = 0; i < numFrames; i++) {
            const frame = player.frames[i] || {}; // Handle potential missing data
            const cell = document.createElement('td');
            // Last frame logic (Standard bowling: 10th frame has 3 throws)
            // Here: The LAST frame has 3 throws if playing full game logic.
            const isLastFrame = (i === numFrames - 1);

            cell.innerHTML = `
                <div class="frame-cell">
                    <div class="frame-throws">
                        <div class="throw-box ${frame.is_strike ? 'strike' : ''}">${frame.throw1 !== undefined && frame.throw1 !== null ? frame.throw1 : ''}</div>
                        <div class="throw-box ${frame.is_spare ? 'spare' : ''}">${frame.throw2 !== undefined && frame.throw2 !== null ? frame.throw2 : ''}</div>
                        ${isLastFrame ? `<div class="throw-box">${frame.throw3 !== undefined && frame.throw3 !== null ? frame.throw3 : ''}</div>` : ''}
                    </div>
                    <div class="frame-score">${frame.score !== null && frame.score !== undefined ? frame.score : ''}</div>
                </div>
            `;
            row.appendChild(cell);
        }

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

    // Update standing pins display using dynamic num_bottles
    const standingPins = state.numBottles - data.pins_knocked;
    document.getElementById('standingPins').textContent =
        data.is_strike || data.frame_complete ? state.numBottles : standingPins;
}

// ==========================================
// Temporal Smoothing Toggle
// ==========================================

function toggleTemporalSmoothing() {
    state.socket.emit('toggle_temporal_smoothing');
}

function updateTemporalSmoothingButton(enabled) {
    const btn = document.getElementById('temporalSmoothingBtn');
    if (btn) {
        if (enabled) {
            btn.textContent = '🔄 Temporal Smoothing: ON';
            btn.classList.add('toggle-active');
        } else {
            btn.textContent = '🔄 Temporal Smoothing: OFF';
            btn.classList.remove('toggle-active');
        }
    }
    showToast(`Temporal Smoothing: ${enabled ? 'ON' : 'OFF'} `, 'info');
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
        `Last throw: ${info.player}, Frame ${info.frame}, Throw ${info.throw} `;
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
        announcement.textContent = `Winner: ${data.winner} ! 🏆`;
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

    // Reset registration state
    state.registrationState = {
        1: { front: false, back: false, left: false, right: false },
        2: { front: false, back: false, left: false, right: false }
    };

    // Reset registration UI
    [1, 2].forEach(num => {
        const card = document.getElementById(`player${num}Card`);
        const status = document.getElementById(`player${num}Status`);
        const regBtn = document.getElementById(`regPlayer${num}Btn`);
        const input = document.getElementById(`player${num}Name`);

        card.classList.remove('registered');
        status.textContent = '';
        regBtn.disabled = true;
        regBtn.textContent = `✅ Register Player ${num} `;
        input.value = '';

        // Reset angle buttons and indicators
        ['front', 'back', 'left', 'right'].forEach(angle => {
            const btn = document.getElementById(`p${num}${capitalize(angle)}Btn`);
            const ind = document.getElementById(`p${num}Ind${capitalize(angle)}`);
            if (btn) {
                btn.disabled = false;
                btn.classList.remove('captured');
                btn.innerHTML = `📷 ${capitalize(angle)} `;
            }
            if (ind) {
                ind.classList.remove('active');
            }
        });

        // Reset registered player dropdown
        const regSelect = document.getElementById(`player${num}RegisteredSelect`);
        if (regSelect) regSelect.selectedIndex = 0;
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
    toast.className = `toast ${type} `;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ==========================================
// Manages Saves Feature
// ==========================================

async function openManageSavesModal() {
    const modal = document.getElementById('manageSavesModal');
    if (modal) {
        modal.classList.add('active');
        // Reset state
        document.getElementById('selectAllSaves').checked = false;
        document.getElementById('deleteSelectedSavesBtn').disabled = true;
        document.getElementById('selectedCount').textContent = '0 selected';

        // Fetch data
        await fetchAndRenderSaves();
    }
}

function closeManageSavesModal() {
    const modal = document.getElementById('manageSavesModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

async function fetchAndRenderSaves() {
    const tbody = document.getElementById('saveListBody');
    tbody.innerHTML = '<tr><td colspan="3" style="text-align: center;">Loading...</td></tr>';

    try {
        const response = await fetch('/api/detailed_players');
        const data = await response.json();
        const players = data.players || [];

        renderSaveList(players);
    } catch (error) {
        console.error('Failed to load saves:', error);
        tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--error);">Failed to load saves.</td></tr>';
    }
}

function renderSaveList(players) {
    const tbody = document.getElementById('saveListBody');
    tbody.innerHTML = '';

    if (players.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No saved players found.</td></tr>';
        return;
    }

    players.forEach(player => {
        const tr = document.createElement('tr');

        const checkboxTd = document.createElement('td');
        checkboxTd.style.textAlign = 'center';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'save-checkbox';
        checkbox.value = player.name;
        checkbox.addEventListener('change', updateDeleteButtonState);
        checkboxTd.appendChild(checkbox);

        const nameTd = document.createElement('td');
        nameTd.textContent = player.name;

        const dateTd = document.createElement('td');
        dateTd.textContent = player.created_at;

        tr.appendChild(checkboxTd);
        tr.appendChild(nameTd);
        tr.appendChild(dateTd);
        tbody.appendChild(tr);
    });
}

function toggleSelectAllSaves() {
    const selectAll = document.getElementById('selectAllSaves');
    const checkboxes = document.querySelectorAll('.save-checkbox');

    checkboxes.forEach(cb => {
        cb.checked = selectAll.checked;
    });

    updateDeleteButtonState();
}

function updateDeleteButtonState() {
    const checkboxes = document.querySelectorAll('.save-checkbox:checked');
    const count = checkboxes.length;
    const btn = document.getElementById('deleteSelectedSavesBtn');
    const countSpan = document.getElementById('selectedCount');

    btn.disabled = count === 0;
    countSpan.textContent = `${count} selected`;

    // Update "Select All" checkbox state if manual selection changed
    const allCheckboxes = document.querySelectorAll('.save-checkbox');
    const selectAll = document.getElementById('selectAllSaves');
    if (allCheckboxes.length > 0 && count === allCheckboxes.length) {
        selectAll.checked = true;
        selectAll.indeterminate = false;
    } else if (count > 0) {
        selectAll.checked = false;
        selectAll.indeterminate = true;
    } else {
        selectAll.checked = false;
        selectAll.indeterminate = false;
    }
}

async function deleteSelectedSaves() {
    const checkboxes = document.querySelectorAll('.save-checkbox:checked');
    const names = Array.from(checkboxes).map(cb => cb.value);

    if (names.length === 0) return;

    if (!confirm(`Are you sure you want to delete ${names.length} player(s)? This cannot be undone.`)) {
        return;
    }

    const btn = document.getElementById('deleteSelectedSavesBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = 'Deleting...';

    try {
        const response = await fetch('/api/delete_players', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ names: names })
        });

        const result = await response.json();

        if (result.success) {
            showToast(`Deleted ${result.deleted.length} players`, 'success');
            if (result.errors && result.errors.length > 0) {
                console.warn('Errors during deletion:', result.errors);
                showToast('Some files could not be deleted', 'warning');
            }

            // Refresh the list in the modal
            await fetchAndRenderSaves();

            // Refresh the main dropdowns (global refresh)
            refreshRegisteredPlayers();

            // Reset selection state
            document.getElementById('selectAllSaves').checked = false;
            document.getElementById('selectAllSaves').indeterminate = false;
            updateDeleteButtonState();

        } else {
            showToast(result.message || 'Deletion failed', 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast('Failed to delete players', 'error');
    } finally {
        btn.disabled = false; // Actually it should be disabled if list is empty or nothing selected, but updateDeleteButtonState checks check boxes.
        // After re-render, no checkboxes are checked, so it will be disabled correctly by `updateDeleteButtonState` call or logic.
        // But wait, `renderSaveList` creates new check boxes which are unchecked.
        // So the button should be disabled.
        btn.innerHTML = originalText;
        updateDeleteButtonState();
    }
}

// ==========================================
// Utility Functions
// ==========================================

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')} `;
}
