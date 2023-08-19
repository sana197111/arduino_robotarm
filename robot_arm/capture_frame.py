import time
import cv2
import numpy as np


def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 캠이 연결되지 않았습니다.")
        return None

    is_success, frame = cap.read()

    if not is_success:
        print("Error: 프레임을 캡처할 수 없습니다.")
    else:
        return frame

    cap.release() 


def find_sticker(color, frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
    elif color == "blue":
        lower = np.array([110, 50, 50])
        upper = np.array([130, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise Exception(f"{color.capitalize()} sticker not found.")

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    return cx, cy


red_positions = []
blue_positions = []


def find_sticker_and_save_coordinates():
    global red_positions, blue_positions
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 캠이 연결되지 않았습니다.")
        return

    while True:
        frame = capture_frame()
        
        if frame is None:  
            break

        red_cx, red_cy = 0, 0  # Add default values for red_cx and red_cy
        blue_cx, blue_cy = 0, 0  # Add default values for blue_cx and blue_cy

        try:
            red_cx, red_cy = find_sticker("red", frame)
            frame = cv2.putText(frame, 'Red', (red_cx + 10, red_cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            red_positions.append((red_cx, red_cy))
            red_found = True
        except Exception as e:
            print(e)
            red_found = False

        try:
            blue_cx, blue_cy = find_sticker("blue", frame)
            frame = cv2.putText(frame, 'Blue', (blue_cx + 10, blue_cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            blue_positions.append((blue_cx, blue_cy))
            blue_found = True
        except Exception as e:
            print(e)
            blue_found = False

        if red_found and blue_found:
            cv2.circle(frame, (red_cx, red_cy), 5, (0, 0, 255), 2)
            cv2.circle(frame, (blue_cx, blue_cy), 5, (255, 0, 0), 2)
        
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break

        print("Red Sticker Coordinate: (%d,%d)" % (red_cx, red_cy))
        print("Blue Sticker Coordinate: (%d,%d)" % (blue_cx, blue_cy))

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    return red_positions, blue_positions