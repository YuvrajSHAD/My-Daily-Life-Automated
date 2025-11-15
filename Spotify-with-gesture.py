### Important notes: You need Spotify Developer Dashboard login + premium of spotify + create the client id and secret on that and you are good to go, make no mistake in URI Callback options, you know once you open it.
### play and pause not refined much, more important for me was next and previous while gaming, studying.

import cv2
import mediapipe as mp
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import ctypes
from collections import deque
import math

# ---------------- Spotify auth (keep your credentials) ----------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="CLIENT ID",
    client_secret="CLIENT SECRET",
    redirect_uri="http://127.0.0.1:8888/callback", 
    scope="user-modify-playback-state user-read-playback-state"
))

# ---------------- Media key fallback setup ----------------
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

def send_media_key(vk_code):
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

# ---------------- Safe Spotify calls (fallbacks included) ----------------
def safe_start_playback(device_id=None, uris=None):
    try:
        kwargs = {}
        if device_id:
            kwargs['device_id'] = device_id
        if uris:
            kwargs['uris'] = uris
        sp.start_playback(**kwargs)
        return True
    except Exception as e:
        send_media_key(VK_MEDIA_PLAY_PAUSE)
        return False

def safe_pause(device_id=None):
    try:
        if device_id:
            sp.pause_playback(device_id=device_id)
        else:
            sp.pause_playback()
        return True
    except Exception:
        send_media_key(VK_MEDIA_PLAY_PAUSE)
        return False

def safe_next(device_id=None):
    try:
        if device_id:
            sp.next_track(device_id=device_id)
        else:
            sp.next_track()
        return True
    except Exception:
        send_media_key(VK_MEDIA_NEXT_TRACK)
        return False

def safe_prev(device_id=None):
    try:
        if device_id:
            sp.previous_track(device_id=device_id)
        else:
            sp.previous_track()
        return True
    except Exception:
        send_media_key(VK_MEDIA_PREV_TRACK)
        return False

# ---------------- MediaPipe / gesture setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- TUNABLE PARAMETERS (make it stricter by raising numbers) ----------------
SMOOTH_WINDOW = 7               # frames required for static gestures (increase to be stricter)
SWIPE_WINDOW = 10               # frames for swipe analysis (increase for longer swipe buffer)
DEBOUNCE_AFTER_ACTION = 0.35    # minimal time between distinct actions

# static gesture stability: hand must be steady (centroid speed)
STATIC_VEL_THRESHOLD = 0.008    # avg centroid speed (normalized units/sec) allowed for static gestures (lower = stricter)

# finger-extension thresholds (normalized distances)
INDEX_EXT_THRESHOLD = 0.11      # index tip vs index MCP distance for "clear" extension
MIDDLE_EXT_THRESHOLD = 0.10     # middle tip vs middle MCP for second finger

# swipe requirements
SWIPE_X_DISPLACEMENT = 0.08     # normalized x displacement threshold across SWIPE_WINDOW
SWIPE_PEAK_VEL = 0.06           # at least one instantaneous x-velocity must exceed this
SWIPE_MAX_Y_DISPLACEMENT = 0.06 # vertical displacement limit to avoid diagonal flails

# buffers
recent_gestures = deque(maxlen=SMOOTH_WINDOW)
centroid_x_buf = deque(maxlen=SWIPE_WINDOW)
centroid_y_buf = deque(maxlen=SWIPE_WINDOW)
time_buf = deque(maxlen=SWIPE_WINDOW)

last_action_time = 0.0
last_fired = None  # 'play','pause','next','prev' or None

# ---------------- helpers ----------------
def distance_points(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_finger_up(tip, pip):
    # more margin to avoid borderline cases
    return tip.y < pip.y - 0.015

def count_extended_fingers(landmarks):
    # returns (count, index_ext_dist, middle_ext_dist, flags)
    index_tip = landmarks[8]; index_mcp = landmarks[5]
    middle_tip = landmarks[12]; middle_mcp = landmarks[9]
    ring_tip = landmarks[16]; ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]; pinky_mcp = landmarks[17]
    thumb_tip = landmarks[4]; thumb_ip = landmarks[3]

    index_up = is_finger_up(index_tip, landmarks[6])  # using PIP for noise resilience
    middle_up = is_finger_up(middle_tip, landmarks[10])
    ring_up = is_finger_up(ring_tip, landmarks[14])
    pinky_up = is_finger_up(pinky_tip, landmarks[18])

    # distances normalized (tip - mcp)
    index_ext_dist = distance_points(index_tip, index_mcp)
    middle_ext_dist = distance_points(middle_tip, middle_mcp)

    return int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up), index_ext_dist, middle_ext_dist, (index_up, middle_up, ring_up, pinky_up), thumb_tip

def hand_centroid(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return sum(xs)/len(xs), sum(ys)/len(ys)

# classify static per-frame gesture candidate
def classify_frame_gesture(landmarks):
    count, idx_dist, mid_dist, flags, thumb_tip = count_extended_fingers(landmarks)
    # require clear tip-to-mcp distances for static gestures to be considered
    if count == 1 and idx_dist >= INDEX_EXT_THRESHOLD:
        return 'play'
    if count == 2 and (idx_dist >= INDEX_EXT_THRESHOLD and mid_dist >= MIDDLE_EXT_THRESHOLD):
        return 'pause'
    return 'none'

# compute average absolute centroid speed (normalized / sec) across the SWIPE window
def compute_avg_centroid_speed():
    if len(centroid_x_buf) < 2:
        return 0.0
    speeds = []
    for i in range(1, len(centroid_x_buf)):
        dx = centroid_x_buf[i] - centroid_x_buf[i-1]
        dy = centroid_y_buf[i] - centroid_y_buf[i-1]
        dt = time_buf[i] - time_buf[i-1]
        if dt <= 0:
            continue
        speeds.append(math.hypot(dx, dy) / dt)
    return sum(speeds)/len(speeds) if speeds else 0.0

# evaluate swipe: returns 'swipe_right','swipe_left' or None
def evaluate_swipe():
    if len(centroid_x_buf) < 2:
        return None
    dx = centroid_x_buf[-1] - centroid_x_buf[0]
    dy = centroid_y_buf[-1] - centroid_y_buf[0]
    dt_total = time_buf[-1] - time_buf[0] if (time_buf[-1] - time_buf[0])>0 else 1e-6
    # instantaneous velocities (x) list
    vx_list = []
    for i in range(1, len(centroid_x_buf)):
        dt = time_buf[i] - time_buf[i-1]
        if dt <= 0:
            continue
        vx_list.append((centroid_x_buf[i] - centroid_x_buf[i-1]) / dt)
    peak_vx = max(vx_list) if vx_list else 0.0
    trough_vx = min(vx_list) if vx_list else 0.0

    # require large x displacement, peak velocity, and limited vertical movement
    if dx > SWIPE_X_DISPLACEMENT and peak_vx >= SWIPE_PEAK_VEL and abs(dy) <= SWIPE_MAX_Y_DISPLACEMENT:
        return 'swipe_right'
    if dx < -SWIPE_X_DISPLACEMENT and trough_vx <= -SWIPE_PEAK_VEL and abs(dy) <= SWIPE_MAX_Y_DISPLACEMENT:
        return 'swipe_left'
    return None

# ---------------- main loop ----------------
device_id = None  # if you want to target your phone, set device_id to the phone's device id

while True:
    ret, frame = cap.read()
    if not ret:
        break
    now = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    candidate_g = 'none'
    cx = cy = None

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        landmarks = handLms.landmark

        candidate_g = classify_frame_gesture(landmarks)

        cx, cy = hand_centroid(landmarks)
        centroid_x_buf.append(cx)
        centroid_y_buf.append(cy)
        time_buf.append(now)

        # swipe evaluation (if candidate still none or we want to override)
        swipe = evaluate_swipe()
        if swipe:
            candidate_g = swipe

        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        centroid_x_buf.clear()
        centroid_y_buf.clear()
        time_buf.clear()
        candidate_g = 'none'

    # smoothing
    recent_gestures.append(candidate_g)
    majority = 'none'
    count_majority = 0
    if recent_gestures:
        majority = max(set(recent_gestures), key=recent_gestures.count)
        count_majority = recent_gestures.count(majority)

    # determine stable gesture candidate
    stable_gesture = None
    if majority in ('play', 'pause'):
        # ensure hand is steady while holding (prevents accidental triggers while adjusting)
        avg_speed = compute_avg_centroid_speed()
        if count_majority >= SMOOTH_WINDOW and avg_speed <= STATIC_VEL_THRESHOLD:
            stable_gesture = majority
    elif majority in ('swipe_right', 'swipe_left'):
        # require fewer frames but a clear swipe (evaluate_swipe already checked)
        if count_majority >= max(3, SMOOTH_WINDOW // 2):
            stable_gesture = majority
    else:
        stable_gesture = 'none'

    # map stable_gesture to action
    action_to_fire = 'none'
    if stable_gesture == 'play':
        action_to_fire = 'play'
    elif stable_gesture == 'pause':
        action_to_fire = 'pause'
    elif stable_gesture == 'swipe_right':
        action_to_fire = 'next'
    elif stable_gesture == 'swipe_left':
        action_to_fire = 'prev'

    # LATCHING / EDGE DETECTION with debounce
    if action_to_fire != 'none':
        if last_fired is None and (now - last_action_time) > DEBOUNCE_AFTER_ACTION:
            if action_to_fire == 'play':
                ok = safe_start_playback(device_id=device_id)
            elif action_to_fire == 'pause':
                ok = safe_pause(device_id=device_id)
            elif action_to_fire == 'next':
                ok = safe_next(device_id=device_id)
            elif action_to_fire == 'prev':
                ok = safe_prev(device_id=device_id)
            last_action_time = now
            last_fired = action_to_fire
    else:
        # no gesture -> clear latch and ready for next
        last_fired = None

    # minimal on-screen debug
    cv2.putText(frame, f"Stable:{stable_gesture} x{count_majority}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Latched:{last_fired}", (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
