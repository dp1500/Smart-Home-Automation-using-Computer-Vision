from flask import Flask, request
from threading import Thread
import threading

app = Flask(__name__)

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

# Variable to store whether an unknown face is detected
global unknown_face_detected
unknown_face_detected = None

@app.route('/update_unknown_face', methods=['POST'])
def update_unknown_face():
    global unknown_face_detected

    # Get the value from the request
    data = request.get_json()

    unknown_face_detected = data.get('status')

    print( "post api call check ", unknown_face_detected)

    return {"message": "unknown status updated succesfully", "unknown_face_detected": unknown_face_detected}

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

def run_flask():
    # Run the Flask app in a separate thread
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    run_flask()
