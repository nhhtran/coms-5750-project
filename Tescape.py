import cv2
import mediapipe as mp
from collections import deque
import copy
import math
import time 
import numpy as np
from model import KeyPointClassifier
import csv
import itertools
from astar import astar
from quadtree_orcld import build_graph, build_tree, find_containing_region
from quadtree_astar import astar as qastar
import threading

class CvFpsCalc(object):
    """
    To calculate FPS of the video feed
    """
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded

# Setting up canvas size to draw on
canvas_width = 2000
canvas_height = 2000
canvas_size = (max(1280, canvas_width), max(720, canvas_height)) 

# Initializing Mediapipe Hands Module
mp_hands = mp.solutions.hands
max_num_hands = 1
min_detection_confidence = 0.5  
min_tracking_confidence = 0.5 
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
    static_image_mode=False,  # Optimization
    model_complexity=1 
)

# Initialize Gesture Classifier
keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

# Define color constants
white = (255, 255, 255)
orange = (0, 165, 255)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
black = (0, 0, 0)

# Initialize variables for GUI
last_click_time = time.time()
selected_color = black
selected_tool = "none"  # Start with no tool selected
pen_x = []
pen_y = []
msg = 'Select a tool to begin'

border_color = (138, 11, 246) 


def disx(pt1, pt2):
    """
    Calculate euclidean distance between two points, p1 and p2
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)

def PasteImg(frame, img, pos):
    """
    Paste an image on frame at specified position
    """
    x, y = pos[0], pos[1]
    try:
        img = cv2.resize(img, (100, 100))
        frame[y:y+100, x:x+100] = img
    except:
        pass 


# Display tools icons
pan = cv2.imread("toolimg/pan.png")
pen = cv2.imread("toolimg/pen.png")
okay = cv2.imread('toolimg/startpoint.jpg')
thumbup = cv2.imread('toolimg/thumbup.png')
thumbdown = cv2.imread('toolimg/thumbdown.png')
altok = cv2.imread('toolimg/endpoint.jpg')
eraser = cv2.imread('toolimg/eraser.jpg')

clip_ok = cv2.imread('toolimg/okay.png')
clip_pinch = cv2.imread('toolimg/pinch.png')

def smooth_points(points_x, points_y, filter_strength=0.7):
    """
    Smoothing points using exponential moving average filter with lower filter strength for better performance
    """
    if len(points_x) <= 2:
        return points_x, points_y
        
    smoothed_x = [points_x[0]]  
    smoothed_y = [points_y[0]]
    
    prev_x, prev_y = points_x[0], points_y[0]
    
    for i in range(1, len(points_x)):
        if points_x[i] == -1:  
            smoothed_x.append(-1)
            smoothed_y.append(-1)
            if i < len(points_x) - 1:  
                prev_x, prev_y = points_x[i+1], points_y[i+1]
            continue
            
        curr_x = filter_strength * prev_x + (1 - filter_strength) * points_x[i]
        curr_y = filter_strength * prev_y + (1 - filter_strength) * points_y[i]
        
        smoothed_x.append(int(curr_x))
        smoothed_y.append(int(curr_y))
        
        prev_x, prev_y = curr_x, curr_y
            
    return smoothed_x, smoothed_y

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cvFpsCalc = CvFpsCalc(buffer_len=5)

# Coordinate history - reduced for better performance
history_length = 16

# Smoothing buffer for pen movement - reduced for performance
pen_smooth_buffer_x = deque(maxlen=3)
pen_smooth_buffer_y = deque(maxlen=3)

# Initialize camera
cap = cv2.VideoCapture(1)
# Camera properties for better performance
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
cap.set(cv2.CAP_PROP_FPS, 30)  
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   


ret, first_frame = cap.read()
if ret:
    original_height, original_width = first_frame.shape[:2]
else:
    original_width, original_height = 1280, 720  

# Scale factor for resizing
scale_factor = 1.0

camera_width = int(original_width * scale_factor)
camera_height = int(original_height * scale_factor)

viewport_size = (camera_width, camera_height)

# Create a larger canvas for panning
offset_x, offset_y = (canvas_size[0] - viewport_size[0]) // 2, (canvas_size[1] - viewport_size[1]) // 2  # Start in center of the canvas

canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

# Panning tool variable
prev_hand_pos = None

# Pen tool variables
stroke_started = False
current_stroke_x = []
current_stroke_y = []

def calc_landmark_list(image, landmarks):
    """
    Convert landmark coordiantes to a matrix
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    """
    Compute relative coordinates and normalize the landmark list
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    """
    Function to draw the detected landmarks and lines connecting these landmarks
    SOURCE: https://github.com/kinivi/hand-gesture-recognition-mediapipe
    """
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def crop_canvas_to_content(canvas, maze_start, maze_end):
    """
    Crops the canvas to only include the area with drawn content.
    """

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    non_zero_pixels = cv2.findNonZero(binary)
    
    if non_zero_pixels is None or len(non_zero_pixels) == 0:
        return canvas, (0, 0)
    
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(canvas.shape[1], w + 2 * margin)
    h = min(canvas.shape[0], h + 2 * margin)

    x_min, x_max = x, x + w
    y_min, y_max = y, y + h
    
    if maze_start[0] < x_min:
        expand_left = x_min - maze_start[0] + margin
        x_min = max(0, maze_start[0] - margin)
        w += expand_left
    elif maze_start[0] >= x_max:
        expand_right = maze_start[0] - x_max + 1 + margin
        w += expand_right
        w = min(canvas.shape[1] - x_min, w)
    
    if maze_start[1] < y_min:
        expand_top = y_min - maze_start[1] + margin
        y_min = max(0, maze_start[1] - margin)
        h += expand_top
    elif maze_start[1] >= y_max:
        expand_bottom = maze_start[1] - y_max + 1 + margin
        h += expand_bottom
        h = min(canvas.shape[0] - y_min, h)
    
    if maze_end[0] < x_min:
        expand_left = x_min - maze_end[0] + margin
        x_min = max(0, maze_end[0] - margin)
        w += expand_left
    elif maze_end[0] >= x_min + w:
        expand_right = maze_end[0] - (x_min + w) + 1 + margin
        w += expand_right
        w = min(canvas.shape[1] - x_min, w)
    
    if maze_end[1] < y_min:
        expand_top = y_min - maze_end[1] + margin
        y_min = max(0, maze_end[1] - margin)
        h += expand_top
    elif maze_end[1] >= y_min + h:
        expand_bottom = maze_end[1] - (y_min + h) + 1 + margin
        h += expand_bottom
        h = min(canvas.shape[0] - y_min, h)
    
    x = int(x_min)
    y = int(y_min)
    w = min(int(w), canvas.shape[1] - x)
    h = min(int(h), canvas.shape[0] - y)
    
    cropped = canvas[y:y+h, x:x+w]
    
    return cropped, (x, y)

# To help user understand hand position on canvas
show_cursor_on_canvas = True 

tools = [
        ("Panning", pan),
        ("close", None),
        ("Draw", pen),
        ("Setting Start Point", okay),
        ("thumb up", thumbup),
        ("thumb down", thumbdown),
        ("Setting End Point", altok),
        ("Eraser", eraser),
    ] 

# Variables for monitoring pen drawing
drawing_delay = 0.5
pen_gesture_start_time = None
pen_drawing_active = False

# Variables for mode changing activation
mode_change_delay = 1.0
thumb_gesture_start_time = None

# Variables for delay in setting start and end points
startend_delay = 0.5
start_point_gesture_start_time = None
end_point_gesture_start_time = None

# Possible Modes
# 0: Drawing Mode
# 1: Maze Solving Mode
mode = 0

# Variables to maintain continuity of strokes
last_stroke_end_time = None 
stroke_gap_threshold = 0.5   
last_stroke_end_position = None

# Determine whether to solve with A* or Quadtree A*
use_astar = None

# To store starting and ending coordiantes for maze
maze_start = None
maze_end = None

# To ensure that once the maze is solved, it doesn't get solved again
maze_solved = False
maze_path = None

# Threading variables
maze_processing_thread = None
maze_computing = False

while cap.isOpened():
    stat, frame = cap.read()
    if not stat:
        print("Error: Couldn't read frame.")
        break
        
    frame = cv2.flip(frame, 1)  
    debug_image = copy.deepcopy(frame)
    fps = cvFpsCalc.get()
    
    height, width, _ = frame.shape
    
    h = int(height * scale_factor)
    w = int(width * scale_factor)
    frame = cv2.resize(frame, (w, h))
    
    viewport_size = (w, h)
    
    # Canvas Window - Stores the drawing
    canvas_display = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
   
    # Message displayed as header
    msg = None
    
    cursor_x, cursor_y = -1, -1

    gesture_id = None
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):

            landmark_list = calc_landmark_list(frame, hand_landmarks)  
            frame = draw_landmarks(frame, landmark_list)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Gesture classification
            gesture_id = keypoint_classifier(pre_processed_landmark_list)
            
            # The following if statements are used to reset gesture variables
            if gesture_id != 4 and gesture_id != 5:
                thumb_gesture_start_time = None
                
            if gesture_id != 6 and gesture_id != 3:
                start_point_gesture_start_time = None
                end_point_gesture_start_time = None
            
            if gesture_id != 2:
                pen_gesture_start_time = None
                pen_drawing_active = False
            
            try:    
                msg = tools[gesture_id][0]
            except:
                msg = None
                
            # Handling of gestures with different functions
            if gesture_id == 2:  
                """
                Gesture : Pointer
                Function : Draw
                """
                if mode == 0:
                    cursor_x, cursor_y = landmark_list[8][0], landmark_list[8][1]
                    
                    current_time = time.time()
                    
                    if pen_gesture_start_time is None:
                        pen_gesture_start_time = current_time
                        pen_drawing_active = False
                        
                        # Connect two strokes if the time gap is small enough
                        if (last_stroke_end_time is not None and 
                            (current_time - last_stroke_end_time) < stroke_gap_threshold and
                            last_stroke_end_position is not None):
                            
                            x = landmark_list[8][0]
                            y = landmark_list[8][1]
                            
                            x_adjusted = x + offset_x
                            y_adjusted = y + offset_y
                            
                            distance = ((x_adjusted - last_stroke_end_position[0])**2 + 
                                    (y_adjusted - last_stroke_end_position[1])**2)**0.5
                            
                            # Connect if strokes are within 50 pixels of each other
                            if distance < 50:  
                                if len(pen_x) > 0 and pen_x[-1] == -1 and pen_y[-1] == -1:
                                    pen_x.pop()
                                    pen_y.pop()
                                    
                                    stroke_started = True
                                    
                                    current_stroke_x = [last_stroke_end_position[0]]
                                    current_stroke_y = [last_stroke_end_position[1]]
                    
                    elapsed_time = current_time - pen_gesture_start_time
                    
                    # To show time until drawing is activated
                    if not pen_drawing_active:
                        remaining = max(0, drawing_delay - elapsed_time)
                        countdown = int(remaining * 10) / 10 
                        
                        canvas_cursor_x = landmark_list[8][0] + offset_x
                        canvas_cursor_y = landmark_list[8][1] + offset_y
                        
                        progress = min(1.0, elapsed_time / drawing_delay)
                        angle = int(360 * (1 - progress))
                        
                        cv2.ellipse(canvas_display, 
                                    (canvas_cursor_x, canvas_cursor_y),
                                    (30,30),
                                    -90, 
                                    0, 
                                    angle,
                                    (0,255,0), 
                                    -1) 
                        
                        cv2.circle(canvas_display, 
                                (canvas_cursor_x, canvas_cursor_y), 
                                30, 
                                (255, 255, 255), 
                                2)
                        
                    
                    if elapsed_time >= drawing_delay:
                        if not pen_drawing_active:
                            pen_drawing_active = True
                        
                        x = landmark_list[8][0]
                        y = landmark_list[8][1]
                        
                        pen_smooth_buffer_x.append(x)
                        pen_smooth_buffer_y.append(y)
                        
                        # Apply smoothing on buffer
                        if len(pen_smooth_buffer_x) >= 3:
                            x = int(sum(pen_smooth_buffer_x) / len(pen_smooth_buffer_x))
                            y = int(sum(pen_smooth_buffer_y) / len(pen_smooth_buffer_y))
                        
                        x_adjusted = x + offset_x
                        y_adjusted = y + offset_y
                        
                        if not stroke_started:
                            stroke_started = True
                            current_stroke_x = [x_adjusted]
                            current_stroke_y = [y_adjusted]
                        else:
                            current_stroke_x.append(x_adjusted)
                            current_stroke_y.append(y_adjusted)
                        
                        pen_x.append(x_adjusted)
                        pen_y.append(y_adjusted)
            
            elif gesture_id == 7:
                """
                Gesture : Yo
                Function : Eraser
                """
                if mode == 0:
                    index_tip_x, index_tip_y = landmark_list[8][0], landmark_list[8][1]
                    cursor_x, cursor_y = index_tip_x, index_tip_y
                    little_tip_x, little_tip_y = landmark_list[20][0], landmark_list[20][1]
                    
                    # Eraser diameter is half the distance between index and little finger tips
                    dist = math.sqrt((index_tip_x - little_tip_x)**2 + (index_tip_y - little_tip_y)**2)
                    eraser_radius = int(dist / 4)
                    
                    # Setting upper and lower limits for radius
                    eraser_radius = max(1, min(eraser_radius, 50))
                    
                    eraser_x = index_tip_x + offset_x
                    eraser_y = index_tip_y + offset_y
                    
                    cv2.circle(canvas_display, (eraser_x, eraser_y), eraser_radius, (200, 200, 200), 2)
                    cv2.circle(canvas_display, (eraser_x, eraser_y), 2, (100, 100, 100), -1)
                    
                    cv2.line(canvas_display, 
                            (eraser_x - eraser_radius + 5, eraser_y), 
                            (eraser_x + eraser_radius - 5, eraser_y), 
                            (150, 150, 150), 1)
                    cv2.line(canvas_display, 
                            (eraser_x, eraser_y - eraser_radius + 5), 
                            (eraser_x, eraser_y + eraser_radius - 5), 
                            (150, 150, 150), 1)
                    
                    if len(pen_x) > 0:
                        new_pen_x = []
                        new_pen_y = []
                        
                        i = 0
                        while i < len(pen_x):
                            if pen_x[i] == -1:
                                new_pen_x.append(pen_x[i])
                                new_pen_y.append(pen_y[i])
                                i += 1
                                continue
                                
                            # Check if point is within eraser radius
                            dist_to_point = math.sqrt((pen_x[i] - eraser_x)**2 + (pen_y[i] - eraser_y)**2)
                            
                            if dist_to_point > eraser_radius:
                                # Point is outside eraser, keep it
                                new_pen_x.append(pen_x[i])
                                new_pen_y.append(pen_y[i])
                            else:
                                # Point is inside eraser - check if we need to split the stroke
                                # If we're erasing a point in the middle of a stroke, we need to add a separator
                                if (i > 0 and pen_x[i-1] != -1 and 
                                    i < len(pen_x)-1 and pen_x[i+1] != -1):
                                    new_pen_x.append(-1)
                                    new_pen_y.append(-1)
                            
                            i += 1

                        pen_x = new_pen_x
                        pen_y = new_pen_y
                        
                    pen_gesture_start_time = None
                    thumb_gesture_start_time = None
                    pen_drawing_active = False
               
            elif gesture_id == 3:
                """
                Gesture : OK
                Function : Set Start Point
                """
                
                cursor_x, cursor_y = -1, -1
                # Check the distance between index tip and thumb tip
                thumb_x = landmark_list[4][0]
                thumb_y = landmark_list[4][1]
                index_x = landmark_list[8][0]
                index_y = landmark_list[8][1]
                
                distance = disx((thumb_x, thumb_y), (index_x, index_y))    
                
                # Only set when distance is lesser than threshold
                if distance < 50:
                    current_time = time.time()
                    
                    if start_point_gesture_start_time is None:
                        start_point_gesture_start_time = current_time
                    
                    elapsed_time = current_time - start_point_gesture_start_time
                    remaining = max(0, startend_delay - elapsed_time)
                    countdown = int(remaining * 10) / 10
                    
                    canvas_cursor_x = landmark_list[8][0] + offset_x
                    canvas_cursor_y = landmark_list[8][1] + offset_y
                    
                    progress = min(1.0, elapsed_time / startend_delay)
                    angle = int(360 * (1 - progress))
                    
                    cv2.ellipse(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y),
                                (30,30),
                                -90, 
                                0, 
                                angle,
                                (0,255,0), 
                                -1)
                    
                    cv2.circle(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y), 
                                30, 
                                (255, 255, 255), 
                                2)
                    
                    if elapsed_time >= startend_delay:
                        if mode == 0:
                            x_index = landmark_list[8][0]
                            y_index = landmark_list[8][1]
                            
                            x_thumb = landmark_list[4][0]
                            y_thumb = landmark_list[4][1]
                            
                            x = (x_index + x_thumb) // 2
                            y = (y_index + y_thumb) // 2
                            
                            x_adjusted = x + offset_x
                            y_adjusted = y + offset_y
                            
                            maze_start = (x_adjusted, y_adjusted)
                        else:
                            if use_astar is not True:
                                maze_solved = False
                                maze_path = None
                            use_astar = True
                            msg = "Solved using A-Star"
                    else:
                        cursor_x = landmark_list[8][0]
                        cursor_y = landmark_list[8][1]
                        
                        if mode == 1: 
                            msg = "Hold to solve using A-Star"
            
            elif gesture_id == 6:
                """
                Gesture : Alt_OK
                Function : Set End Point
                """
                
                cursor_x, cursor_y = -1, -1
                # Check the distance between index tip and thumb tip
                thumb_x = landmark_list[4][0]
                thumb_y = landmark_list[4][1]
                
                index_x = landmark_list[8][0]
                index_y = landmark_list[8][1]
                
                distance = disx((thumb_x, thumb_y), (index_x, index_y))
                
                # Only set when distance is lesser than threshold
                if distance < 50:
                    current_time = time.time()
                    
                    if end_point_gesture_start_time is None:
                        end_point_gesture_start_time = current_time
                        
                    elapsed_time = current_time - end_point_gesture_start_time
                    remaining = max(0, startend_delay - elapsed_time)
                    countdown = int(remaining * 10) / 10
                    
                    canvas_cursor_x = landmark_list[8][0] + offset_x
                    canvas_cursor_y = landmark_list[8][1] + offset_y
                    
                    progress = min(1.0, elapsed_time / startend_delay)
                    angle = int(360 * (1 - progress))
                    
                    cv2.ellipse(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y),
                                (30,30),
                                -90, 
                                0, 
                                angle,
                                (0,255,0), 
                                -1)
                    
                    cv2.circle(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y), 
                                30, 
                                (255, 255, 255), 
                                2)
                    
                    if elapsed_time >= startend_delay:
                        if mode == 0:
                            x_index = landmark_list[8][0]
                            y_index = landmark_list[8][1]
                            
                            x_thumb = landmark_list[4][0]
                            y_thumb = landmark_list[4][1]
                            
                            x = (x_index + x_thumb) // 2
                            y = (y_index + y_thumb) // 2
                            
                            x_adjusted = x + offset_x
                            y_adjusted = y + offset_y
                            
                            maze_end = (x_adjusted, y_adjusted)
                        else:
                            if use_astar is not False:
                                maze_solved = False
                                maze_path = None
                            use_astar = False
                            msg = "Solved using Quadtree"
                    else:
                        cursor_x = landmark_list[8][0]
                        cursor_y = landmark_list[8][1]  
                        if mode == 1: 
                            msg = "Hold to solve using Quadtree"       
                    
            elif gesture_id == 0:
                """
                Gesture : Open
                Function : Panning
                """
                cursor_x, cursor_y = -1, -1
                
                # Use palm center for panning
                palm_x = (landmark_list[0][0] + landmark_list[9][0]) // 2
                palm_y = (landmark_list[0][1] + landmark_list[9][1]) // 2
                
                if prev_hand_pos is not None:
                    dx = palm_x - prev_hand_pos[0]
                    dy = palm_y - prev_hand_pos[1]
                    
                    new_offset_x = max(0, min(canvas_size[0] - viewport_size[0], offset_x - dx))
                    new_offset_y = max(0, min(canvas_size[1] - viewport_size[1], offset_y - dy))
                    
                    offset_x = new_offset_x
                    offset_y = new_offset_y
                    
                prev_hand_pos = (palm_x, palm_y)
            elif gesture_id == 4:
                """
                Gesture : Thumb Up
                Function : Change Mode Up
                """
                cursor_x, cursor_y = -1, -1
                
                if mode == 1:
                    msg = "Cannot change mode up"
                elif maze_start is None or maze_end is None:
                    msg = "Set start and end points first"
                else:    
                    current_time = time.time()
                    msg = f"Hold to change mode to [Mode {mode+1}]"
                    
                    if thumb_gesture_start_time is None:
                        thumb_gesture_start_time = current_time
                    
                    elapsed_time = current_time - thumb_gesture_start_time
                    remaining = max(0, mode_change_delay - elapsed_time)
                    countdown = int(remaining * 10) / 10
                    
                    canvas_cursor_x = landmark_list[9][0] + offset_x
                    canvas_cursor_y = landmark_list[9][1] + offset_y
                    
                    progress = min(1.0, elapsed_time / mode_change_delay)
                    angle = int(360 * (1 - progress))
                    
                    cv2.ellipse(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y),
                                (30,30),
                                -90, 
                                0, 
                                angle,
                                (0,255,0), 
                                -1)
                    
                    cv2.circle(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y), 
                                30, 
                                (255, 255, 255), 
                                2)
                    
                    if elapsed_time >= mode_change_delay:
                        mode = mode+1
                        thumb_gesture_start_time = None
                    
            elif gesture_id == 5:
                """
                Gesture : Thumb Down
                Function : Change Mode Down
                """
                cursor_x, cursor_y = -1, -1
                
                if mode == 0:
                    msg = "Cannot change mode down"
                else:
                    current_time = time.time()
                    msg = f"Hold to change mode to [Mode {mode-1}]"
                    
                    if thumb_gesture_start_time is None:
                        thumb_gesture_start_time = current_time
                    
                    elapsed_time = current_time - thumb_gesture_start_time
                    remaining = max(0, mode_change_delay - elapsed_time)
                    countdown = int(remaining * 10) / 10
                    
                    canvas_cursor_x = landmark_list[9][0] + offset_x
                    canvas_cursor_y = landmark_list[9][1] + offset_y
                    
                    progress = min(1.0, elapsed_time / mode_change_delay)
                    angle = int(360 * (1 - progress))
                    
                    cv2.ellipse(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y),
                                (30,30),
                                -90, 
                                0, 
                                angle,
                                (0,255,0), 
                                -1)
                    
                    cv2.circle(canvas_display,
                                (canvas_cursor_x, canvas_cursor_y), 
                                30, 
                                (255, 255, 255), 
                                2)
                    
                    if elapsed_time >= mode_change_delay:
                        mode = mode-1
                        thumb_gesture_start_time = None
               
            else:
                """
                Gesture : Closed
                Function : -- No action --
                """
                msg = None
                
                cursor_x = landmark_list[8][0]
                cursor_y = landmark_list[8][1]
                
                if mode == 0:
                    pen_gesture_start_time = None
                    thumb_gesture_start_time = None
                    pen_drawing_active = False
                    
                    # End a stroke if it has already started
                    if stroke_started:
                        pen_x.append(-1)
                        pen_y.append(-1)
                        stroke_started = False
                        
                        # Apply smoothing to the completed stroke
                        if len(current_stroke_x) > 2:
                            smooth_x, smooth_y = smooth_points(current_stroke_x, current_stroke_y)
                            
                            stroke_start = len(pen_x) - len(current_stroke_x) - 1
                            if stroke_start < 0:
                                stroke_start = 0
                                
                            pen_x[stroke_start:len(pen_x)-1] = smooth_x
                            pen_y[stroke_start:len(pen_y)-1] = smooth_y
                
                if gesture_id != 0 or selected_tool != "pan":
                    prev_hand_pos = None
                
                # Clear smoothing buffers
                pen_smooth_buffer_x.clear()
                pen_smooth_buffer_y.clear()

    else:
        """
            No hand detected by mediapipe
        """
        if mode == 0:
            if stroke_started:
                # Store the last position and time before ending the stroke
                current_time = time.time()
                last_stroke_end_time = current_time
                
                # Get the last valid point from the current stroke
                if len(current_stroke_x) > 0 and len(current_stroke_y) > 0:
                    last_stroke_end_position = (current_stroke_x[-1], current_stroke_y[-1])
                
                pen_x.append(-1)
                pen_y.append(-1)
                stroke_started = False
                
                # Apply smoothing to the completed stroke 
                if len(current_stroke_x) > 2:
                    smooth_x, smooth_y = smooth_points(current_stroke_x, current_stroke_y)
                    
                    stroke_start = len(pen_x) - len(current_stroke_x) - 1
                    if stroke_start < 0:
                        stroke_start = 0
                        
                    pen_x[stroke_start:len(pen_x)-1] = smooth_x
                    pen_y[stroke_start:len(pen_y)-1] = smooth_y
            
            prev_hand_pos = None
    
    # Display FPS for debugging
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
    
    if len(pen_x) > 0:
        for i in range(len(pen_x) - 1):
            if pen_x[i] == -1 or pen_x[i+1] == -1:
                continue
            cv2.line(canvas_display, (pen_x[i], pen_y[i]), (pen_x[i+1], pen_y[i+1]), selected_color, 3)
    
    maze_canvas = canvas_display.copy()
    
    if maze_start:
        cv2.circle(canvas_display, maze_start, 5, green, -1)
        cv2.circle(canvas_display, maze_start, 2, black, -1)
        
    if maze_end:
        cv2.circle(canvas_display, maze_end, 5, red, -1)
        cv2.circle(canvas_display, maze_end, 2, black, -1)

    # Draw cursor on canvas for debugging alignment
    if show_cursor_on_canvas and cursor_x != -1 and cursor_y != -1:
        cursor_canvas_x = cursor_x + offset_x
        cursor_canvas_y = cursor_y + offset_y
        if 0 <= cursor_canvas_x < canvas_size[0] and 0 <= cursor_canvas_y < canvas_size[1]:
            cv2.circle(canvas_display, (cursor_canvas_x, cursor_canvas_y), 5, red, -1)
    
    if mode == 1:
        # Find the path from start to end and draw it
        
        # Entity                ||  Variable Name
        # --------------------- ||  ----------------
        # Drawing Canvas        ||  canvas_display
        # Starting Coordinate   ||  maze_start
        # Ending Coordinate     ||  maze_end
        
        if not maze_solved and use_astar is not None:
            cropped_canvas, (crop_x, crop_y) = crop_canvas_to_content(maze_canvas, maze_start, maze_end)
            cropped_start = (maze_start[0] - crop_x, maze_start[1] - crop_y)
            cropped_end = (maze_end[0] - crop_x, maze_end[1] - crop_y)
            
            maze_gray = cv2.cvtColor(cropped_canvas, cv2.COLOR_BGR2GRAY)
            _, binary_canvas = cv2.threshold(maze_gray, 10, 255, cv2.THRESH_BINARY)
            binary_maze = (binary_canvas == 0).astype(np.uint8)
            
            # ==== For A-Star Implementation ====
            if use_astar is True:
                obstacles = set()
                obstacles.add(1)
                (cropped_path, closed) = astar(binary_maze, obstacles, cropped_start, cropped_end)
            elif use_astar is False:
            # ==== For Quadtree Implementation ====                
                depth = math.ceil(math.log2(max(binary_canvas.shape[0] - 1, binary_canvas.shape[1] - 1)))
                quadtree = build_tree(binary_canvas, cropped_start, cropped_end, depth, binary_canvas.shape[1] - 1, binary_canvas.shape[0] - 1)  
                graph = build_graph(quadtree)

                start_cord = find_containing_region(quadtree, cropped_start)
                end_cord = find_containing_region(quadtree, cropped_end)

                if start_cord is not None and end_cord is not None:
                    cropped_path, closed = qastar(graph, start_cord, end_cord)
                    cropped_path = [cropped_start] + cropped_path + [cropped_end]
                
            try:
                maze_path = [(pt[0] + crop_x, pt[1] + crop_y) for pt in cropped_path]
                maze_solved = True
            except:
                pass
        else:
            if maze_path is not None:
                for i in range(len(maze_path) - 1):
                    cv2.line(canvas_display, maze_path[i], maze_path[i + 1], (255, 0, 0), 3)
    
    else:
        maze_solved = False
        maze_path = None
        use_astar = None
    
    # Extract viewport from canvas
    try:
        view_y_end = min(offset_y + viewport_size[1], canvas_size[1])
        view_x_end = min(offset_x + viewport_size[0], canvas_size[0])
        
        if view_y_end <= offset_y:
            view_y_end = offset_y + 1
        if view_x_end <= offset_x:
            view_x_end = offset_x + 1
            
        # Extract the visible portion of canvas
        viewport = canvas_display[offset_y:view_y_end, offset_x:view_x_end].copy()
        
        # Resize viewport to match camera frame dimensions
        if viewport.shape[0] != viewport_size[1] or viewport.shape[1] != viewport_size[0]:
            viewport = cv2.resize(viewport, (viewport_size[0], viewport_size[1]))
        
    except Exception as e:
        print(f"Viewport error: {e}")
        viewport = cv2.resize(canvas_display, viewport_size)
    
    combined_display = frame.copy()
    alpha = 0.6
    
    combined_display = cv2.addWeighted(viewport, alpha, frame, 1-alpha, 0)
    title = "TraceEscape"
    font_scale, thickness = 1.6, 5
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(combined_display, title, ((width - tw) // 2, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, red, thickness)
    
    if msg is not None:
        if mode == 0 or (mode != 0 and (gesture_id==0 or gesture_id == 4 or gesture_id == 5 or gesture_id == 3 or gesture_id == 6)):    
            cv2.putText(combined_display, msg, (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, black, 2)
            if mode==0: PasteImg(combined_display, tools[gesture_id][1], (20, 565))

    if mode == 1:
        # display clip_ok with transparent background with text saying "hello" below it at the middle right of the frame 
        clip_ok = cv2.imread("toolimg/okay.png", cv2.IMREAD_UNCHANGED)
        clip_ok = cv2.resize(clip_ok, (int(clip_ok.shape[1] * 0.3), int(clip_ok.shape[0] * 0.3)), interpolation=cv2.INTER_AREA)
        clip_ok_height, clip_ok_width = clip_ok.shape[:2]
        clip_ok_x = width - clip_ok_width - 20
        clip_ok_y = height - clip_ok_height - camera_height//2 + 50
        overlay = combined_display[clip_ok_y:clip_ok_y + clip_ok_height, clip_ok_x:clip_ok_x + clip_ok_width]
        text = "A-Star"
        text_x = clip_ok_x + (clip_ok_width - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]) // 2
        text_y = clip_ok_y + clip_ok_height + 30
        cv2.putText(combined_display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
        mask = clip_ok[:, :, 3] / 255.0
        overlay = overlay * (1 - mask[:, :, np.newaxis]) + clip_ok[:, :, :3] * mask[:, :, np.newaxis]
        combined_display[clip_ok_y:clip_ok_y + clip_ok_height, clip_ok_x:clip_ok_x + clip_ok_width] = overlay
                
        
        clip_pinch = cv2.imread("toolimg/pinch.png", cv2.IMREAD_UNCHANGED)
        clip_pinch = cv2.resize(clip_pinch, (int(clip_pinch.shape[1] * 0.3), int(clip_pinch.shape[0] * 0.3)), interpolation=cv2.INTER_AREA)
        clip_pinch_height, clip_pinch_width = clip_pinch.shape[:2]
        clip_pinch_x = 20
        clip_pinch_y = height - clip_pinch_height - camera_height//2 + 50
        overlay = combined_display[clip_pinch_y:clip_pinch_y + clip_pinch_height, clip_pinch_x:clip_pinch_x + clip_pinch_width]
        text = "Quadtree"
        text_x = clip_pinch_x + (clip_pinch_width - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]) // 2
        text_y = clip_pinch_y + clip_pinch_height + 30
        cv2.putText(combined_display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
        mask = clip_pinch[:, :, 3] / 255.0
        overlay = overlay * (1 - mask[:, :, np.newaxis]) + clip_pinch[:, :, :3] * mask[:, :, np.newaxis]
        combined_display[clip_pinch_y:clip_pinch_y + clip_pinch_height, clip_pinch_x:clip_pinch_x + clip_pinch_width] = overlay
        
        
    mode_text = f"Mode {mode}"
    mode_font_scale, mode_thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, mode_font_scale, mode_thickness)
    cv2.putText(combined_display, mode_text, (width - tw - 20, height - th - 10), cv2.FONT_HERSHEY_SIMPLEX, mode_font_scale, red, mode_thickness)
    
    cv2.imshow('TraceEscape', combined_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()