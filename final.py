import numpy as np
import mediapipe as mp
import time
import pyautogui # Using pyautogui for mouse and keyboard
import cv2
import math

# --- Hand Detector Class (using mediapipe) ---
class handDetector():
    def _init_(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, trackCon=0.6): # Adjusted defaults
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = [] # Initialize lmList here

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = [] # Reset list for current frame
        bbox = []
        if self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks) > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList = []
            yList = []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                # if draw: # Optional: draw landmarks if needed for debugging
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if xList and yList:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        # Return empty list and bbox if no hand found
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            return [0, 0, 0, 0, 0]

        # Thumb: Compare x-coordinate of tip to x-coordinate of PIP joint
        # Assumes a relatively standard hand orientation (modify if needed)
        # Use lmList[4][1] (tip x) and lmList[3][1] (joint x)
        # This check is for a hand facing the camera; adjust if hand is sideways
        if len(self.lmList) > 4: # Ensure landmarks exist
             # Simple check for right hand facing camera: tip x < pip x
             # For left hand facing camera: tip x > pip x
             # A more robust check might compare to lmList[2] or based on handedness if available
             if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]: # Simple right hand logic
                 fingers.append(1)
             else:
                 fingers.append(0)
        else:
             fingers.append(0) # Default if not enough landmarks

        # Other 4 Fingers: Compare y-coordinate of tip to y-coordinate of PIP joint (lower y is higher up)
        for id in range(1, 5):
             if len(self.lmList) > self.tipIds[id] and len(self.lmList) > self.tipIds[id] - 2: # Ensure landmarks exist
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
             else:
                 fingers.append(0) # Default if not enough landmarks

        # Ensure fingers list always has 5 elements
        while len(fingers) < 5:
            fingers.append(0)

        return fingers[:5] # Return only the first 5 elements

    def findDistance(self, p1, p2, img, draw=True, r=10, t=3):
        lineInfo = [0, 0, 0, 0, 0, 0] # Default value
        length = 0 # Default value
        if len(self.lmList) > max(p1, p2):
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
                cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            lineInfo = [x1, y1, x2, y2, cx, cy]

        return length, img, lineInfo

    def isThumbDown(self):
        # Basic check: Thumb tip (4) y-coord is significantly below thumb CMC joint (1) y-coord
        # And other fingers are down
        if len(self.lmList) > 4:
            thumb_tip_y = self.lmList[4][2]
            thumb_cmc_y = self.lmList[1][2] # Carpometacarpal joint
            thumb_mcp_y = self.lmList[2][2] # Metacarpophalangeal joint

            # Thumb tip needs to be below MCP joint by a threshold
            # Adjust threshold as needed
            vertical_diff_threshold = 15
            if thumb_tip_y > thumb_mcp_y + vertical_diff_threshold:
                 return True
        return False

# --- Configuration ---
wCam, hCam = 1280, 720
frameR = 150 # Frame reduction for operational area
smoothening = 5 # Smoothing factor for cursor movement
click_threshold = 45 # Distance between index and middle finger for click
press_debounce_duration = 0.5 # Min seconds between key presses
toggle_debounce_duration = 1.0 # Min seconds between mode toggles

# --- Initialization ---
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)

detector = handDetector(maxHands=1) # Use our class
wScr, hScr = pyautogui.size()
print(f"Screen Size: {wScr} x {hScr}")

pyautogui.FAILSAFE = True
# pyautogui.PAUSE = 0.01 # Optional small pause

# --- State Variables ---
cursor_mode_on = False # Start with cursor mode OFF
last_press_time = 0
last_toggle_time = 0
last_action_description = "None" # For display

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        time.sleep(0.5) # Wait a bit before retrying
        continue

    img = cv2.flip(img, 1) # Flip for intuitive control

    # 1. Find Hand Landmarks
    img = detector.findHands(img, draw=True) # Draw hand skeleton
    lmList, bbox = detector.findPosition(img, draw=False) # Get landmark list, bbox optional

    fingers = [0, 0, 0, 0, 0] # Default
    current_time = time.time()

    # Only process if landmarks are found
    if len(lmList) != 0:
        fingers = detector.fingersUp()
        # print(f"Fingers: {fingers}") # Debug

        # --- Gesture Detection & Action ---

        # A. Check for HIGH FIVE (Toggle Mode) - Prioritize this check
        is_high_five = (fingers == [1, 1, 1, 1, 1])
        if is_high_five and (current_time - last_toggle_time > toggle_debounce_duration):
            cursor_mode_on = not cursor_mode_on # Toggle the mode
            last_toggle_time = current_time
            last_action_description = f"Mode Toggled: {'ON' if cursor_mode_on else 'OFF'}"
            print(f"--- Cursor Mode {'ENABLED' if cursor_mode_on else 'DISABLED'} ---")
            time.sleep(0.1) # Small pause after toggle

        # B. Execute based on Mode (and avoid action if toggle just happened)
        elif (current_time - last_toggle_time > 0.2): # Add small buffer after toggle

            # B.1 CURSOR MODE ON - Mouse Control
            if cursor_mode_on:
                last_action_description = "Cursor Mode Active"
                # Index finger tip for position
                if len(lmList) > 8:
                    x1, y1 = lmList[8][1:]

                    # Check if ONLY index finger is up (Moving Mode)
                    if fingers == [0, 1, 0, 0, 0]:
                        last_action_description = "Cursor: Moving"
                        # Convert Coordinates
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                        # Smoothen Values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        # Move Mouse
                        move_x = max(0, min(wScr - 1, clocX))
                        move_y = max(0, min(hScr - 1, clocY))
                        try:
                            pyautogui.moveTo(move_x, move_y)
                        except pyautogui.FailSafeException:
                            print("FailSafe triggered! Exiting.")
                            break
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    # Check if Index and Middle fingers are up (Clicking Mode)
                    elif fingers == [0, 1, 1, 0, 0] and len(lmList) > 12:
                         last_action_description = "Cursor: Checking Click"
                         # Find distance between fingers
                         length, img, lineInfo = detector.findDistance(8, 12, img, r=8, t=2) # Smaller visuals
                         # Click mouse if distance short
                         if length < click_threshold:
                            # Ensure lineInfo is valid before drawing circle
                            if len(lineInfo) >= 6 and lineInfo[4] !=0 and lineInfo[5] !=0:
                                 cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                            try:
                                pyautogui.click()
                                last_action_description = "Cursor: CLICKED"
                                print("--- CLICK ---")
                                # Debounce click itself slightly if needed
                                time.sleep(0.2) # Prevent rapid multi-clicks
                            except pyautogui.FailSafeException:
                                print("FailSafe triggered! Exiting.")
                                break

            # B.2 CURSOR MODE OFF - Gesture to Key Press
            else:
                # Check for key press gestures only if debounce time has passed
                if (current_time - last_press_time > press_debounce_duration):
                    action_taken = False
                    key_to_press = None

                    # Check specific finger counts
                    if fingers == [0, 1, 0, 0, 0]: key_to_press = '1'
                    elif fingers == [0, 1, 1, 0, 0]: key_to_press = '2'
                    elif fingers == [0, 1, 1, 1, 0]: key_to_press = '3'
                    elif fingers == [0, 1, 1, 1, 1]: key_to_press = '4'
                    # Check Thumb gestures (ensure they don't overlap with finger counts)
                    elif fingers == [1, 0, 0, 0, 0]: key_to_press = 'q'
                    # Check Thumb Down (requires fingers down and thumb low)
                    elif fingers == [1, 1, 0, 0, 0]: key_to_press = 'e'
                    # Check Fist (do nothing)
                    elif fingers == [0, 0, 0, 0, 0]:
                         last_action_description = "Gesture: Fist"
                         pass # Explicitly do nothing for a fist

                    # Press key if a gesture was recognized
                    if key_to_press:
                        try:
                            pyautogui.press(key_to_press)
                            action_taken = True
                            last_action_description = f"Gesture: Pressed '{key_to_press}'"
                            print(f"--- Pressed '{key_to_press}' ---")
                        except pyautogui.FailSafeException:
                             print("FailSafe triggered! Exiting.")
                             break

                    # Update last press time if an action was taken
                    if action_taken:
                        last_press_time = current_time
                # else: # Optional: Indicate waiting for debounce
                #      if not cursor_mode_on and any(f == 1 for f in fingers) and fingers != [1,1,1,1,1]:
                #            last_action_description = "Gesture: Debouncing"
                #      elif not cursor_mode_on and fingers == [0,0,0,0,0]:
                #           last_action_description = "Gesture: Fist" # Keep showing fist

    else:
        # No hand detected
        last_action_description = "No Hand Detected"


    # --- Display Information ---
    # Draw operational frame
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Display Mode Status
    mode_text = f"CURSOR MODE: {'ON' if cursor_mode_on else 'OFF'}"
    mode_color = (0, 255, 0) if cursor_mode_on else (0, 0, 255)
    cv2.putText(img, mode_text, (20, hCam - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

    # Display Last Action
    cv2.putText(img, f"Last Action: {last_action_description}", (20, hCam - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

    # Show Image
    cv2.imshow("Gesture Control", img)

    # Exit Key
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # Use a different key like ESC to quit if 'q' is used for gestures
        print("Escape key pressed. Exiting.")
        break
    # Example using ESC key to quit:
    # key = cv2.waitKey(1) & 0xFF
    # if key == 27: # 27 is the ASCII code for Escape
    #      print("Escape key pressed. Exiting.")
    #      break


# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()