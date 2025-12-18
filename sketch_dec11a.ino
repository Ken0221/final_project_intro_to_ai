#include <Servo.h>

Servo myServo;
const int servoPin = 9; // 你的接線是 A0
String inputString = ""; // 用來暫存收到的數據

void setup() {
  Serial.begin(9600);    // 設定通訊速率 (必須跟 Python 一致)
  myServo.attach(servoPin);
  myServo.write(90);     // 預設先轉到中間
  Serial.println("Arduino Ready"); // 告訴 Python 我準備好了
}

void loop() {
  // 檢查是否有資料從 Python 傳過來
  while (Serial.available()) {
    // 讀取傳過來的整數 (角度)
    int angle = Serial.parseInt(); 
    
    // 讀取換行符號，確認指令結束
    if (Serial.read() == '\n') {
      // 限制角度在 0~180 之間保護馬達
      angle = constrain(angle, 0, 180);
      
      myServo.write(angle); // 執行動作！
      
      // 回傳確認訊息給 Python (方便除錯)
      Serial.print("Moved to: ");
      Serial.println(angle);
    }
  }
}