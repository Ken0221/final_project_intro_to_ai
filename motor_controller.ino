/*
 * Motor Controller for Camera Tracking System
 * =============================================
 * 
 * This Arduino sketch receives pan/tilt commands from Python
 * and controls two MG996R servo motors for camera tracking.
 * 
 * Hardware Setup:
 * - Pin 9 (D9):  Pan Servo  - MG996R 360° Continuous Rotation
 *                            (Controls speed: 90=stop, <90=CCW, >90=CW)
 * - Pin 10 (D10): Tilt Servo - MG996R 180° Standard
 *                            (Controls position: 0-180 degrees)
 * 
 * Communication Protocol:
 * - Baud Rate: 115200
 * - Format: <PanSpeed,TiltAngle>
 * - Example: <75,90> = Pan rotating CCW at medium speed, Tilt at center
 * 
 * Author: AI Intro Final Project
 * Date: 2024-12
 */

#include <Servo.h>

// ============== Pin Definitions ==============
const int PAN_SERVO_PIN = 9;    // D9 - 360° continuous rotation
const int TILT_SERVO_PIN = 10;  // D10 - 180° standard servo

// ============== Servo Objects ==============
Servo panServo;   // Continuous rotation (speed control)
Servo tiltServo;  // Standard servo (position control)

// ============== Communication Settings ==============
const long BAUD_RATE = 115200;    // Must match Python script
const char START_MARKER = '<';
const char END_MARKER = '>';

// ============== Buffer for incoming data ==============
const byte MAX_MESSAGE_LENGTH = 32;
char receivedChars[MAX_MESSAGE_LENGTH];
boolean newData = false;

// ============== Current values ==============
int currentPanSpeed = 90;    // 90 = stopped (for continuous rotation)
int currentTiltAngle = 90;   // 90 = center position

// ============== Safety limits ==============
const int PAN_MIN = 0;
const int PAN_MAX = 180;
const int TILT_MIN = 0;
const int TILT_MAX = 180;

// ============== Debugging ==============
const boolean DEBUG_MODE = true;

void setup() {
  // Initialize Serial communication
  Serial.begin(BAUD_RATE);
  
  // Attach servos to pins
  panServo.attach(PAN_SERVO_PIN);
  tiltServo.attach(TILT_SERVO_PIN);
  
  // Set initial positions
  // Pan: 90 = stopped (continuous rotation servo interprets this as no movement)
  // Tilt: 90 = center position (looking straight ahead)
  panServo.write(currentPanSpeed);
  tiltServo.write(currentTiltAngle);
  
  // Signal ready
  if (DEBUG_MODE) {
    Serial.println("Motor Controller Ready");
    Serial.print("Pan Servo on Pin ");
    Serial.print(PAN_SERVO_PIN);
    Serial.println(" (360° Continuous)");
    Serial.print("Tilt Servo on Pin ");
    Serial.print(TILT_SERVO_PIN);
    Serial.println(" (180° Standard)");
    Serial.print("Baud Rate: ");
    Serial.println(BAUD_RATE);
    Serial.println("Waiting for commands in format: <PanSpeed,TiltAngle>");
    Serial.println("=========================================");
  }
}

void loop() {
  // Check for incoming data
  receiveData();
  
  // Process received data
  if (newData) {
    processCommand();
    newData = false;
  }
}

/**
 * Receive data with start and end markers
 * Format: <PanSpeed,TiltAngle>
 */
void receiveData() {
  static boolean receiveInProgress = false;
  static byte index = 0;
  char rc;
  
  while (Serial.available() > 0) {
    rc = Serial.read();
    
    if (receiveInProgress) {
      if (rc != END_MARKER) {
        receivedChars[index] = rc;
        index++;
        
        // Prevent buffer overflow
        if (index >= MAX_MESSAGE_LENGTH) {
          index = MAX_MESSAGE_LENGTH - 1;
        }
      } else {
        // End marker received
        receivedChars[index] = '\0';  // Null terminate
        receiveInProgress = false;
        index = 0;
        newData = true;
      }
    } else if (rc == START_MARKER) {
      receiveInProgress = true;
      index = 0;
    }
  }
}

/**
 * Parse and execute the received command
 * Expected format: PanSpeed,TiltAngle
 */
void processCommand() {
  char* token;
  int panSpeed;
  int tiltAngle;
  
  // Parse pan speed (first value before comma)
  token = strtok(receivedChars, ",");
  if (token != NULL) {
    panSpeed = atoi(token);
    
    // Parse tilt angle (second value after comma)
    token = strtok(NULL, ",");
    if (token != NULL) {
      tiltAngle = atoi(token);
      
      // Apply commands
      setPanSpeed(panSpeed);
      setTiltAngle(tiltAngle);
      
      // Debug output
      if (DEBUG_MODE) {
        Serial.print("CMD: Pan=");
        Serial.print(currentPanSpeed);
        Serial.print(" (");
        printPanDirection(currentPanSpeed);
        Serial.print("), Tilt=");
        Serial.print(currentTiltAngle);
        Serial.println("°");
      }
    }
  }
}

/**
 * Set pan servo speed (continuous rotation)
 * 
 * For MG996R 360° continuous rotation servo:
 * - 90: Stop
 * - 0-89: Rotate counter-clockwise (speed increases as value decreases)
 * - 91-180: Rotate clockwise (speed increases as value increases)
 */
void setPanSpeed(int speed) {
  // Constrain to valid range
  speed = constrain(speed, PAN_MIN, PAN_MAX);
  
  currentPanSpeed = speed;
  panServo.write(speed);
}

/**
 * Set tilt servo angle (standard positional servo)
 * 
 * For MG996R 180° servo:
 * - 0: One extreme position
 * - 90: Center position
 * - 180: Other extreme position
 */
void setTiltAngle(int angle) {
  // Constrain to valid range
  angle = constrain(angle, TILT_MIN, TILT_MAX);
  
  currentTiltAngle = angle;
  tiltServo.write(angle);
}

/**
 * Helper function to print pan direction for debugging
 */
void printPanDirection(int speed) {
  if (speed < 85) {
    Serial.print("CCW");
  } else if (speed > 95) {
    Serial.print("CW");
  } else {
    Serial.print("STOP");
  }
}

/**
 * Emergency stop - stops both servos
 * Can be called if needed for safety
 */
void emergencyStop() {
  panServo.write(90);   // Stop pan rotation
  tiltServo.write(90);  // Center tilt
  currentPanSpeed = 90;
  currentTiltAngle = 90;
  
  if (DEBUG_MODE) {
    Serial.println("EMERGENCY STOP ACTIVATED");
  }
}
