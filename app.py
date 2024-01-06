from flask import Flask, request, Response, jsonify
from threading import Thread
import threading
import cv2


import firebase_admin
from firebase_admin import credentials, messaging

app = Flask(__name__)

# Initialize Firebase Admin SDK with your credentials
cred = credentials.Certificate("smart_home_automation_2\home-automation-f4916-firebase-adminsdk-eu45p-bf29f14315.json")
firebase_admin.initialize_app(cred)


def send_notification(title, body):
    try:
        
        push_message = messaging.Message(
            data={
                'title': title,
                'body': body,
            },
            topic="notify",  # You can use a topic or specify the `token` directly
            # token = "ehb54P3WSGKyOaV-EfWXkX:APA91bFKQhBLsEmQ0iIRRpT__bz47yXXUBXcpti6u1rWeHAAud-bnqLdJSM4VssrnEtiMC-OpMguLsraac2JD7dSpPpmBSCspLjJ9JIbgXCgzOyI1TPaoVSeWdRQqhk7A4ZgDlfnrO1A"
        )

        print(push_message)

        # Send the message
        response = messaging.send(push_message)
        print(response)

        # Return a standard Python dictionary instead of using jsonify
        return {"success": True, "response": response}
    
    except Exception as e:
        # Return a standard Python dictionary instead of using jsonify
        return {"success": False, "error": str(e)}

send_noti = send_notification("Intruder Detected!", "Possible intruder detected. Open the app to view details")
print(send_noti)

# Shared variable among different parts of your application
last_detected_position = "unknown" # updated every 80-100 ms


# Lock for thread safety
api_lock = threading.Lock()

@app.route('/update_position', methods=['POST'])
def update_position():
    global last_detected_position
    data = request.get_json()
    position = data.get('position')
    with api_lock:
        last_detected_position = position
    return {"message": "Position updated successfully"}, 200

@app.route('/get_last_position', methods=['GET'])
def get_last_position():
    global last_detected_position

    code = -1

    if last_detected_position == "left":
        code = 0
    elif last_detected_position == "right":
        code = 1
    else:
        code = -1

        
    with api_lock:
        return {"position_code": code
                ,"last_detected_position": last_detected_position}, 200

def update_position_async(position):
    # Function to update position asynchronously
    update_thread = Thread(target=update_position, args=(), kwargs={"position": position})
    update_thread.start()


# /////// Face Detetction API ./////////////

# Adding family member 

# from flask import Flask, request, jsonify
import os

# app = Flask(__name__)




import datetime 

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        # Check if the 'image' file is present in the request
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"})

        image_file = request.files['image']
        image_name = request.form['name']
        

        static = 'C:/Users/DEVANSH/Desktop/ANPR/FLOW YLO/smart_home_automation_2/static/family_images/'

        # Generate a unique filename based on current timestamp
        # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # image_filename = f"uploaded_image_{timestamp}.jpg"
        image_filename = f"{image_name}.jpg"
        # Save the image to the static folder or process it as needed
        image_file.save(os.path.join(static, image_filename))

        return {"success": True, "message": "Image uploaded successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# if __name__ == '__main__':
#     app.run(debug=True)


# Variable to store whether an unknown face is detected
global unknown_face_detected
unknown_face_detected = None


"""

[
    {
    "id":"",
    "timestamp": "",
    "image_url": ""
    }

]

two apis:

first: for latest intruder detected PendingDeprecationWarning

second: for all the intruder pics since the first notification/alert """

global intruder_list 
intruder_list = []

@app.route('/update_unknown_face', methods=['POST'])
def update_unknown_face():
    # global unknown_face_detected
    global intruder_list 

    # Get the value from the request
    data = request.get_json()

    intruder_data = data.get('intruder_data')
    # timestamp = data.get('timestamp')

    intruder_list.append(intruder_data)


    unknown_face_detected = data.get('unknown_face_detected', False)

    # Check if notification data is included
    notification_data = data.get('notification_data')
    if notification_data:
        # Extract notification details and send the notification
        title = notification_data.get('title')
        body = notification_data.get('body')
        # image_path = notification_data.get('image_path')
        
        # Call the function to send the notification
        send_notification(title, body)

    return {"message": "unknown status updated successfully", "unknown_face_detected": unknown_face_detected}

#////// API to get latest intruder photo////

@app.route('/get_latest_intruder', methods=['GET'])
def get_latest_intruder():

    global intruder_list
    latest_intruder = intruder_list.pop()
    return {"latest_intruder_data": latest_intruder}

from flask import send_from_directory

@app.route('/image/<image_name>', methods=['GET'])
def image(image_name):
    unknown_faces_directory = r"C:\Users\DEVANSH\Desktop\ANPR\FLOW YLO\smart_home_automation_2\static\unknown_faces"

    # Use send_from_directory to serve the image
    return send_from_directory(unknown_faces_directory, image_name)

    

# def update_unknown_face():
#     global unknown_face_detected

#     # Get the value from the request
#     data = request.get_json()

#     unknown_face_detected = data.get('status')

#     print( "post api call check ", unknown_face_detected)

#     return {"message": "unknown status updated succesfully", "unknown_face_detected": unknown_face_detected}







@app.route('/get_unknown_face_status', methods=['GET'])
def get_unknown_face_status():
    # global unknown_face_detected

    if unknown_face_detected == False:
        return {"unknown_face_detected" : "Family member detected",
                "intruder_status": 0}
    elif unknown_face_detected == True:
        return {"unknown_face_detected": "an unknown face has been detected",
                "intruder_status": 1}
        # return {"unknown_face_detected" : "No face detected"}
    else:
        return {"unknown_face_detected": unknown_face_detected,
                "intruder_status": -1}
    


#//////////////////// Fire Detection Part ////////

# Variable to store whether fire is detected 
fire_detected_status = None
fire_lock = threading.Lock()

@app.route('/update_fire_status', methods=['POST'])
def update_fire_status():
    global fire_detected_status

    # Get the value from the request
    data = request.get_json()

    # Update fire detection status
    with fire_lock:
        fire_detected_status = data.get('status')

    return {"message": "Fire status updated successfully", "fire_detected_status": fire_detected_status}

@app.route('/get_fire_status', methods=['GET'])
def get_fire_status():
    with fire_lock:
        return {"fire_detected_status": fire_detected_status}
    

#//////////////////// Fire Detection Part END //////// 





@app.route('/', methods=['GET'])
def server_test():
    return "Server is running", 200

#////////////// video feed live code //////////// 

# camera = cv2.VideoCapture(0) \
    



# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#////////////// video feed live code END ////////////

def run_flask():
    # Run the Flask app in a separate thread
    app.run(debug=True, port=5000)

if __name__ == '__main__': 
    run_flask()



