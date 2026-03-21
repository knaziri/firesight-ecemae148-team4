import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        hsv_frame = param
        h, s, v   = hsv_frame[y, x]
        print(f"Pixel ({x}, {y}) -> H:{h} S:{s} V:{v}")

print("Move mouse over colors to see HSV values.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("HSV Feed", hsv)
    cv2.setMouseCallback("Camera Feed", mouse_callback, hsv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
