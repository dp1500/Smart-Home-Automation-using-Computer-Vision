# Smart Home Automation with Object Detection and Face Recognition

## Overview

This project implements a smart home automation system that incorporates object detection and classification using YOLO (You Only Look Once) model, along with face recognition using Facebook's Face Recognition API. The system is designed to detect intruders and fires, and send real-time alerts to users via a Flutter mobile application.

## Features

- **Peron and Fire Detection**: Utilizes the YOLO model to detect and classify objects, including persons and fires, in live video streams. The detection algorithm processes live video frames captured from the camera and applies the YOLO model to identify objects within the frames. Bounding boxes are drawn around detected objects for visualization that can be streamed live to the user through a front-end application.
<img src="https://github.com/dp1500/Football-Match-Prediction/blob/main/simulation%20table.jpg" alt="Result Table" width="300" height="300">

- **Intruder Detection using Face Recognition**: Implements Facebook's Face Recognition API to identify intruders by comparing detected faces with known family member faces. The system maintains a database of known family member faces. When a new face is detected in the video stream, it is compared against the database using the Face Recognition API. If a match is found, the person is identified as a family member; otherwise, they are classified as an intruder.
<img src="https://github.com/dp1500/Football-Match-Prediction/blob/main/simulation%20table.jpg" alt="Result Table" width="300" height="300">

- **Real-time Alerts**: Sends notifications to users via a Flutter app when intruders or fires are detected. The Flask backend server communicates with the Flutter app via RESTful APIs. When an intruder or fire is detected, the server sends a push notification to the Flutter app using the Firebase Admin SDK. The app receives the notification and displays an alert to the user along with frames(images) with detections, providing real-time updates on potential security threats.

- **Live Stream using WebSocket Server**: Streams live video feed from the camera to connected clients using WebSocket for real-time monitoring. The WebSocket server script establishes a bidirectional communication channel between the server and clients, allowing the server to push live video frames to connected clients in real time. Clients, such as the Flutter app or a web browser, receive the video feed and display it to the user for remote monitoring of the premises.

- **Flask Backend**: Implements a Flask server to handle communication between the detection algorithms and the Flutter app. The Flask server exposes RESTful APIs that enable the detection scripts to update the position of detected objects, upload family member images for face recognition, and send notifications to the Flutter app. The server processes incoming requests, performs necessary operations, and returns appropriate responses to the clients.
- 

## Technologies Used

- **YOLO (You Only Look Once)**: A state-of-the-art, real-time object detection system for identifying objects in images and video streams.

- **OpenCV (Open Source Computer Vision Library)**: Used for capturing video streams from the camera, image processing, and drawing bounding boxes around detected objects.

- **Facebook Face Recognition API**: Utilized for facial recognition tasks to identify known family members from detected faces.

- **Flask**: Python-based micro web framework used to develop the backend server for handling API requests and communication with the Flutter app.

- **REST API**: Implemented RESTful endpoints in the Flask backend to facilitate communication between the client (Flutter app) and server. The REST API handles various operations such as retrieving data, updating resources, and triggering actions.

- **WebSocket**: A communication protocol that enables bidirectional communication between clients and servers over a single, long-lived connection.

- **Firebase Admin SDK**: Integrated to send push notifications to the Flutter app when intruders or fires are detected.

- **Multi-Threading**: Multithreading is employed within the Flask backend for multiple APIs to enhance server performance and responsiveness by creating and handling concurrent tasks efficiently.
  
- **CUDA**: (Compute Unified Device Architecture): Employed for GPU acceleration, CUDA technology enables parallel computing on NVIDIA GPUs, enhancing the performance of deep learning tasks such as object detection.

## Contributors

- [Your Name] - Project Lead & Developer
- [Contributor 1] - Backend Developer
- [Contributor 2] - Frontend Developer

