import serial
import time

# --- 設定區 ---
COM_PORT = 'COM4'   # 請確認你的 Port 是 COM4 (依照你之前的截圖)
BAUD_RATE = 9600    # 必須跟 Arduino 設定的一樣
# -------------

try:
    print(f"正在連接 {COM_PORT}...")
    # 建立連接
    arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    
    # 【關鍵】Arduino 剛連上時會自動重啟，必須等待 2 秒讓它準備好
    # 如果不等待，第一筆指令通常會失敗
    time.sleep(2)
    print("連接成功！開始控制...")
    
    # 清空一開始的緩衝區雜訊
    arduino.reset_input_buffer()

    while True:
        # 讓使用者輸入角度
        user_input = input("請輸入角度 (0-180) 或輸入 'q' 離開: ")
        
        if user_input.lower() == 'q':
            print("程式結束")
            break
            
        try:
            # 檢查輸入是否為數字
            angle = int(user_input)
            
            if 0 <= angle <= 180:
                # 1. 組合指令：數字 + 換行符號 (\n)
                # Arduino 的 Serial.parseInt() 依靠非數字字元(如 \n) 來判斷數字結束
                command = f"{angle}\n"
                
                # 2. 編碼並發送
                arduino.write(command.encode('utf-8'))
                print(f"-> 發送: {angle} 度")
                
                # 3. (選用) 讀取 Arduino 的回傳確認，確認它真的轉了
                time.sleep(0.1) # 稍微等一下 Arduino 處理
                while arduino.in_waiting:
                    response = arduino.readline().decode().strip()
                    if response:
                        print(f"<- Arduino 回覆: {response}")
            else:
                print("錯誤：請輸入 0 到 180 之間的數字")
                
        except ValueError:
            print("錯誤：請輸入有效的整數")

except serial.SerialException:
    print("無法連接 Arduino！")
    print("1. 請檢查 USB 是否插好")
    print("2. 請確認 COM Port 是否正確")
    print("3. **請關閉 Arduino IDE 的 Serial Monitor**")

except Exception as e:
    print(f"發生未預期的錯誤: {e}")

finally:
    # 確保程式結束時關閉連線
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("連線已關閉")