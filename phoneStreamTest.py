import cv2

vid_stream_URL = 'http://192.168.1.46:4747/video'
cap = cv2.VideoCapture(vid_stream_URL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame.")
        break

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Create a window with adjustable dimensions
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL) 

    # Adjust the window size based on frame dimensions
    cv2.resizeWindow('Object Detection', width, height)

    # Your image processing and object detection code here
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
