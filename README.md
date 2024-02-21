# Smart Home Automation with Object Detection and Face Recognition

## Overview

This project implements a smart home automation system that incorporates object detection and classification using YOLO (You Only Look Once) model, along with face recognition using Facebook's Face Recognition API. The system is designed to detect intruders and fires, and send real-time alerts to users via a Flutter mobile application.

## Features

- **Object Detection**: Utilizes YOLO model to detect and classify objects, including persons and fires, in live video streams. The YOLO model is trained on a large dataset of images containing various objects and their classes. The detection algorithm processes live video frames captured from the camera and applies the YOLO model to identify objects within the frames. Bounding boxes are drawn around detected objects for visualization.

- **Face Recognition**: Implements Facebook's Face Recognition API to identify intruders by comparing detected faces with known family member faces. The system maintains a database of known family member faces, and when a new face is detected in the video stream, it is compared against the database using the Face Recognition API. If a match is found, the person is identified as a family member; otherwise, they are classified as an intruder.

- **Real-time Alerts**: Sends notifications to users via a Flutter app when intruders or fires are detected. The Flask backend server communicates with the Flutter app via RESTful APIs. When an intruder or fire is detected, the server sends a push notification to the Flutter app using the Firebase Admin SDK. The app receives the notification and displays an alert to the user, providing real-time updates on potential security threats.

- **Flask Backend**: Implements a Flask server to handle communication between the detection algorithms and the Flutter app. The Flask server exposes RESTful APIs that enable the detection scripts to update the position of detected objects, upload family member images for face recognition, and send notifications to the Flutter app. The server processes incoming requests, performs necessary operations, and returns appropriate responses to the clients.

- **WebSocket Server**: Streams live video feed from the camera to connected clients using WebSocket for real-time monitoring. The WebSocket server script establishes a bidirectional communication channel between the server and clients, allowing the server to push live video frames to connected clients in real-time. Clients, such as the Flutter app or a web browser, receive the video feed and display it to the user for remote monitoring of the premises.

## Technologies Used

- **YOLO (You Only Look Once)**: A state-of-the-art, real-time object detection system for identifying objects in images and video streams.

- **OpenCV (Open Source Computer Vision Library)**: Used for capturing video streams from the camera, image processing, and drawing bounding boxes around detected objects.

- **Facebook Face Recognition API**: Utilized for facial recognition tasks to identify known family members from detected faces.

- **Flask**: Python-based micro web framework used to develop the backend server for handling API requests and communication with the Flutter app.

- **WebSocket**: A communication protocol that enables bidirectional communication between clients and servers over a single, long-lived connection.

- **Firebase Admin SDK**: Integrated to send push notifications to the Flutter app when intruders or fires are detected.

## Project Structure

The project consists of three main components:

1. **Object Detection and Face Recognition Script**: Implements the core functionality of the system. Utilizes YOLO model for object detection, Facebook's Face Recognition API for face recognition, and OpenCV for video stream processing.

2. **Flask Server and APIs**: Provides a Flask-based backend server to handle communication between the detection script and the Flutter app. Includes APIs for updating the position of detected objects, uploading family member images, and sending notifications.

3. **WebSocket Server Script**: Sets up a WebSocket server to stream live video feed from the camera to connected clients in real-time, facilitating remote monitoring.

## Usage

1. **Clone Repository**: Clone the project repository to your local machine.

2. **Install Dependencies**: Install the necessary Python dependencies using `pip install -r requirements.txt`.

3. **Run Flask Server**: Execute the Flask server script to start the backend server for handling API requests.

4. **Start WebSocket Server**: Run the WebSocket server script to initiate the live video feed streaming.

5. **Run Object Detection Script**: Execute the object detection and face recognition script to begin detecting intruders and fires in the video feed.

6. **View Alerts**: Receive real-time alerts and notifications on the Flutter app when intruders or fires are detected.

## Contributors

- [Your Name] - Project Lead & Developer
- [Contributor 1] - Backend Developer
- [Contributor 2] - Frontend Developer

## License

This project is licensed under the [MIT License](LICENSE).
