import cv2

def check_cameras():
    print("Checking camera indices 0-5...")
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[SUCCESS] Camera index {i} is working. Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"[WARNING] Camera index {i} opened but failed to read frame.")
            cap.release()
        else:
            print(f"[FAIL] Camera index {i} could not be opened.")

if __name__ == "__main__":
    check_cameras()
