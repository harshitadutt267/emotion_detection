import cv2
from deepface import DeepFace
import json
import random

# Load content recommendations from JSON file
try:
    with open('content_recommendations.json', 'r') as file:
        content_data = json.load(file)
except FileNotFoundError:
    print("Error: 'content_recommendations.json' file not found.")
    exit()

def get_recommendation(emotion):
    """Fetches a random music, joke, and video recommendation for the detected emotion."""
    try:
        music = random.choice(content_data['music'].get(emotion, ["No music recommendation available"]))
        joke = random.choice(content_data['jokes'].get(emotion, ["No joke available"]))
        video = random.choice(content_data['videos'].get(emotion, ["No video available"]))
        return music, joke, video
    except KeyError:
        return "No recommendation available", "No recommendation available", "No recommendation available"

# Load a more accurate face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
print("Face cascade loaded successfully")

# Start capturing video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

failed_detection_count = 0  # Track consecutive frames with no faces
MAX_FAILED_DETECTIONS = 20  # Stop if no face is detected for 20 frames
FRAME_SKIP = 5  # Analyze every 5th frame to improve efficiency
frame_count = 0

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue  # Skip frames for efficiency

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame (adjust parameters for better accuracy)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))
        print(f"Detected faces: {len(faces)}")

        if len(faces) == 0:
            failed_detection_count += 1
            if failed_detection_count > MAX_FAILED_DETECTIONS:
                print("No face detected for too long. Exiting...")
                break
        else:
            failed_detection_count = 0  # Reset count when a face is detected

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Analyze the face to predict emotions
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                music, joke, video = get_recommendation(emotion)
                print(f"Detected Emotion: {emotion}")
                print(f"Music Recommendation: {music}")
                print(f"Joke Recommendation: {joke}")
                print(f"Video Recommendation: {video}")

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

    except Exception as e:
        print(f"Exception occurred: {e}")
        break

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User exited the program.")
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

'''
import cv2
from deepface import DeepFace
import json
import random
from collections import deque

# Load content recommendations from JSON file
try:
    with open('content_recommendations.json', 'r') as file:
        content_data = json.load(file)
except FileNotFoundError:
    print("Error: 'content_recommendations.json' file not found.")
    exit()

def get_recommendation(emotion):
    """Fetches a random music, joke, and video recommendation for the detected emotion."""
    try:
        music = random.choice(content_data['music'].get(emotion, ["No music recommendation available"]))
        joke = random.choice(content_data['jokes'].get(emotion, ["No joke available"]))
        video = random.choice(content_data['videos'].get(emotion, ["No video available"]))
        return music, joke, video
    except KeyError:
        return "No recommendation available", "No recommendation available", "No recommendation available"

# Load an improved face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
print("Face cascade loaded")

# Start capturing video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

failed_detection_count = 0  # Track consecutive frames with no faces
MAX_FAILED_DETECTIONS = 20  # Stop if no face is detected for 20 frames
emotion_history = deque(maxlen=10)  # Store last 10 detected emotions

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert frame to grayscale and apply preprocessing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)  # Normalize brightness
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))
        print(f"Detected faces: {len(faces)}")

        if len(faces) == 0:
            failed_detection_count += 1
            if failed_detection_count > MAX_FAILED_DETECTIONS:
                print("No face detected for too long. Exiting...")
                break
        else:
            failed_detection_count = 0  # Reset count when a face is detected

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))  # Normalize face size

            # Analyze the face to predict emotions
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                detected_emotion = result[0]['dominant_emotion']
                emotion_history.append(detected_emotion)
                emotion = max(set(emotion_history), key=emotion_history.count)  # Most frequent emotion
                music, joke, video = get_recommendation(emotion)
                print(f"Detected Emotion: {emotion}")
                print(f"Music Recommendation: {music}")
                print(f"Joke Recommendation: {joke}")
                print(f"Video Recommendation: {video}")

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

    except Exception as e:
        print(f"Exception occurred: {e}")
        break

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User exited the program.")
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

'''