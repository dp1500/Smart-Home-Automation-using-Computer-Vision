

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from ultralytics import YOLO
import cv2
import cvzone
import math
import requests  # Add this import for API communication
import time
import torch
import threading
from concurrent.futures import ThreadPoolExecutor

import face_recognition

# import sys
# sys.path.append('..') 

from app import *

# import app

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
 
# Use a ThreadPoolExecutor with a max number of threads
thread_pool = ThreadPoolExecutor(max_workers=5)


# Time-based check for API call
last_api_call_time = time.time()
api_call_interval = 1.0  # Set the desired interval in seconds


# Define the width of the room (adjust according to your room dimensions)
room_width = 1280
 
# cap = cv2.VideoCapture(0)  # For Webcam
# cap = cv2.VideoCapture("smart_home_automation_2\classroom_vid1.mp4") # For Video
cap = cv2.VideoCapture("smart_home_automation_2\classroom_vid4.mp4") # For shorter Video

# for phone camera stream:

# vid_stream_URL = 'http://192.168.1.46:4747/video'
# cap = cv2.VideoCapture(vid_stream_URL)  

cap.set(3, 1280) 
cap.set(4, 720)
 
model = YOLO("../Yolo-Weights/yolov8l.pt")
model.to('cuda')
torch.cuda.set_device(0)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0
 
# ////////////////////// API ///////////////////////

# Global variable to store the last detected position
last_detected_position = None

# api_lock = threading.Lock()

# Flask server endpoint
flask_server_endpoint = "http://127.0.0.1:5000"  # Update with your Flask server's actual address

# Function to send position information to the Flask server in a separate thread
def send_position_to_flask_server_async(position):
    with api_lock:
        url = f"{flask_server_endpoint}/update_position"
        payload = {"position": position}
        response = requests.post(url, json=payload)
        print("API Response:", response.text)


# ///////////////////////  Image detection and recoginition setup ///////////

# Load a sample image of the family members (replace with your own images)
family_member_image_paths = [r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\family_images\devansh.jpg", r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\family_images\nitin.jpg"]
known_face_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img_path))[0] for img_path in family_member_image_paths]


# Function to send position information to the Flask server in a separate thread
# def unknown_detected_api_call(status):
#     with api_lock:
#         url = f"{flask_server_endpoint}/update_unknown_face"
#         payload = {"status": status}
#         response = requests.post(url, json=payload)
#         print("API Response:", response.text)

import datetime 
# Function to send intruder detection status with notification
def unknown_detected_api_call(status, image_filename=None):
    print("levelllll  ahahahahhahaahahahhahahahhahhahahahahahhahahahahhahahahhahahahhahahhah")
    if status:
        with api_lock:
            url = f"{flask_server_endpoint}/update_unknown_face"

            # Include additional data for notification in the payload
            timestamp = datetime.datetime.utcnow().isoformat()

            intruder_data = {
                "timestamp": timestamp,
                "image_name": image_filename,
            }

            payload = {
                "Alert_type": "intruder",
                "status": "true",
                "intruder_data": intruder_data,
                "notification_data": {
                    'title': 'Intruder Detected!',
                    'body': 'Possible intruder detected. Open the app to view details.',
                }
            }

            response = requests.post(url, json=payload)
            print("API Response:", response.text)




# # Directory paths for saving the images
static_images_directory = r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\images"
static_faces_directory = r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\faces"
static_unknown_faces_directory = r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\unknown_faces"


def recognize_face(frame):
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any of the known family members
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            # Face matched with a family member
            print("Face recognized as a family member!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # threading.Thread(target=unknown_detected_api_call, args=(False,)).start()
            # Add your alert logic here
                        # Draw a bounding box around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Extract the cropped image of the face
            cropped_face = frame[top:bottom, left:right]

            # Save the extracted face to the /static/faces/ directory
            face_filename = f"{static_faces_directory}/known_face_{time.time()}.jpg"
            cv2.imwrite(face_filename, cropped_face)

            # Save the whole frame (with the bounding box) to the /static/images/ directory
            image_filename = f"{static_images_directory}/known_face_detected_image_{time.time()}.jpg" 
            cv2.imwrite(image_filename, frame)
            # threading.Thread(target=unknown_detected_api_call, args=(False,image_filename)).start()
            
        else:
            # Face did not match with any family member
            print("Unknown face detected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # threading.Thread(target=unknown_detected_api_call, args=(True,)).start()
            # threading.Thread(target=unknown_detected_api_call, args=(image_filename,)).start()

            # # Draw a bounding box around the detected face
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Increase the size of the bounding box by a certain factor (e.g., 20%)
            expand_factor = 0.2  # You can adjust this factor based on your requirements
            expanded_left = max(0, int(left - expand_factor * (right - left)))
            expanded_top = max(0, int(top - expand_factor * (bottom - top)))
            expanded_right = min(frame.shape[1], int(right + expand_factor * (right - left)))
            expanded_bottom = min(frame.shape[0], int(bottom + expand_factor * (bottom - top)))

            # Draw the rectangle with the expanded bounding box
            cv2.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), (0, 0, 255), 2)


            # Extract the cropped image of the face
            cropped_face = frame[top:bottom, left:right]

            # Save the extracted face to the /static/faces/ directory
            face_filename = f"{static_unknown_faces_directory}/unknown_face_detected_only_face{time.time()}.jpg"
            cv2.imwrite(face_filename, cropped_face)

            current_time = time.time()
            # Save the whole frame (with the bounding box) to the /static/images/ directory
            image_filename = f"{static_unknown_faces_directory}/unknown_face_detected_scene_image{current_time}.jpg" #complete picture of the scene
            image_name = f"unknown_face_detected_scene_image{current_time}.jpg"
            # print("unknown_detected_api_call ka callllllllll")
            cv2.imwrite(image_filename, frame)

            # Call the function with both status and image_filename
            threading.Thread(target=unknown_detected_api_call, args=(True, image_name)).start()

            

            # Display the cropped face
            # cv2.imshow("Detected Face", cropped_face)

    # Draw the detected faces on the original frame
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the original frame with bounding boxes
    cv2.imshow("Object Detection", frame)


# /////////////////////////// Setup End /////////////////


# //////////////////// Fire setup //////////////////////

fire_model = YOLO("smart_home_automation_2\smoke_best.pt")

static_fire_directory = r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\fire"

# Function to send fire status to the Flask server in a separate thread
def send_fire_status_to_flask_server_async(image_name):
    with api_lock:

        print("fire was detected fire was sent to api (api should be called ) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print(image_name)

        url = f"{flask_server_endpoint}/update_fire_status"
        payload = {"status": "fire detected"}

        # Include additional data for notification in the payload
        timestamp = datetime.datetime.utcnow().isoformat()

        fire_data = {
            "timestamp": timestamp,
            "image_name": image_name,
        }

        payload = {
            "Alert_type": "fire",
            "status": "true",
            "fire_data": fire_data,
            "notification_data": {
                'title': 'Fire Detected!',
                'body': 'Possible Fire detected. Open the app to view details.',
            }
        }

        response = requests.post(url, json=payload)
        print("API Response:", response.text)
        

# Function to save image with bounding boxes
def save_image_with_bounding_boxes(image, boxes, save_path):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(save_path, image)

# Function to detect fire in the frame
def detect_fire(img):
    results = fire_model(img, show=False)
    fire_detected = False
    fire_boxes = []

    for r in results:
        if r.boxes.conf is not None:  # This will ensure that id is not None
            # print(r.boxes.id.cpu().numpy().astype(int))
            print(" level 1 id change")

        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            fire_boxes.append((x1, y1, x2, y2))

        if len(boxes) > 0:
            fire_detected = True

    return fire_detected, fire_boxes

import os

# Function to send fire status, save the image, and update the Flask server
def process_fire_detection(img):
    fire_detected, fire_boxes = detect_fire(img)

    if fire_detected:

        print("fire was detected by the model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        current_time = time.time()
        save_path = f"{static_fire_directory}/fire_detected_image_{current_time}.jpg"
        image_name = f"fire_detected_image_{current_time}.jpg"

        print(image_name)

        save_image_with_bounding_boxes(img, fire_boxes, save_path)



        # Send fire status to the Flask server in a separate thread
        threading.Thread(target=send_fire_status_to_flask_server_async, args=(image_name,)).start()

    return fire_detected
# //////////////////// Smoke setup END  //////////////////////

frame_counter = 0  # Initialize the frame counter

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # //// fire detection /////
    fire_detected = process_fire_detection(img)
    #/// fire detection end ///////

    
    results = model(img, stream=True)
    for r in results:

        if r.boxes.conf is not None: # this will ensure that id is not None
            # print(r.boxes.id.cpu().numpy().astype(int)) 
            print("level 2 id change")

        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
 
            if cls != 0:
                break

            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.6:
                # Calculate the center of the detected person
                person_center_x = (x1 + x2) // 2
                person_center_y = (y1 + y2) // 2
                
                # Time-based check for API call
                current_time = time.time()

            
                # Determine left or right side based on the x-coordinate
                if person_center_x < room_width // 2:
                    cv2.putText(img, "Left Side", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # send_position_to_api("left")
                    # if current_time - last_api_call_time >= api_call_interval:
                    threading.Thread(target=send_position_to_flask_server_async, args=("left",)).start()
                else:
                    cv2.putText(img, "Right Side", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # send_position_to_api("right")
                    # if current_time - last_api_call_time >= api_call_interval:
                    threading.Thread(target=send_position_to_flask_server_async, args=("right",)).start()
                    

                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                if frame_counter % 10 == 0:
                    recognize_face(img)

                frame_counter += 1  # Increment the frame counter
            
            

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Object Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
