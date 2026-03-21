import cv2
import numpy as np
from itertools import combinations

# --- Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

print("Running ArUco detector. Press 'q' or Escape to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        # Store center positions keyed by marker ID
        centers = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_corners in enumerate(corners):
                c = marker_corners[0]
                marker_id = ids[i][0]

                # --- Center ---
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                centers[marker_id] = (cx, cy)

                # --- Orientation ---
                dx = c[1][0] - c[0][0]
                dy = c[1][1] - c[0][1]
                angle_deg = np.degrees(np.arctan2(dy, dx))

                # --- Heading arrow ---
                arrow_length = 50
                arrow_tip = (
                    int(cx + arrow_length * np.cos(np.radians(angle_deg))),
                    int(cy + arrow_length * np.sin(np.radians(angle_deg)))
                )
                cv2.arrowedLine(frame, (cx, cy), arrow_tip, (0, 255, 0), 2, tipLength=0.3)

                # --- Center dot ---
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # --- ID and angle text ---
                cv2.putText(frame, f"ID: {marker_id}",
                    (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Angle: {angle_deg:.1f} deg",
                    (cx + 10, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- Draw distances between all detected marker pairs ---
            if len(centers) >= 2:
                for (id_a, id_b) in combinations(centers.keys(), 2):
                    pt_a = centers[id_a]
                    pt_b = centers[id_b]

                    # Pixel distance
                    dist = np.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)

                    # Line between the two markers
                    cv2.line(frame, pt_a, pt_b, (255, 255, 0), 1)

                    # Label at midpoint
                    mid_x = int((pt_a[0] + pt_b[0]) / 2)
                    mid_y = int((pt_a[1] + pt_b[1]) / 2)
                    cv2.putText(frame, f"{dist:.1f}px",
                        (mid_x + 5, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        else:
            cv2.putText(frame, "No markers detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Marker count ---
        count = len(centers)
        cv2.putText(frame, f"Markers detected: {count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("ArUco Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty("ArUco Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("Stopping.")
finally:
    cap.release()
    cv2.destroyAllWindows()