import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime, timedelta
import pyttsx3

# Initialize the TTS engine 
engine = pyttsx3.init()

# Open the video capture device
video_capture = cv2.VideoCapture(0)

# Load known faces
sai_image = face_recognition.load_image_file("C:\\Python_project\\Images\\002.JPG")
sai_encoding = face_recognition.face_encodings(sai_image)[0]

vish_image = face_recognition.load_image_file("C:\\Python_project\\Images\\003.JPG")
vish_encoding = face_recognition.face_encodings(vish_image)[0]

known_face_encodings = [sai_encoding, vish_encoding]
known_face_names = ["Sai Nivedha", "Vishnu Paavani"]

# Set to keep track of recorded attendance and the last time they were marked
attendance_dict = {}

# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Open the csv file for writing
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])  # Write the header row

# Start the infinite loop to process frames
while True:
    # Read a frame from the video capture device
    _, frame = video_capture.read()
    
    # Resize the frame to 1/4 of its original size for faster processing 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Process each face detected in the frame
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index].upper()  # Make the name uppercase
            
            # Add the text if a person is present
            font = cv2.FONT_HERSHEY_COMPLEX  # Use a different font
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (0, 0, 255)
            thickness = 3
            lineType = 3
            cv2.putText(frame, f"WELCOME {name}", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType) 
            
            # Get the current time
            current_time = datetime.now()
            
            # Check if the attendance has already been recorded recently
            if name in attendance_dict:
                last_recorded_time = attendance_dict[name]
                if current_time - last_recorded_time < timedelta(minutes=2):
                    engine.say(f"Attendance already marked for {name}")
                    engine.runAndWait()
                    continue
            
            # Read out the welcome message and mark the attendance
            engine.say(f"Welcome {name}")
            engine.runAndWait()
            
            attendance_dict[name] = current_time
            lnwriter.writerow([name, current_time.strftime("%H:%M:%S")])
                    
    # Display the processed frame with text annotations
    cv2.imshow("Camera", frame)
    
    # Check for the "q" key to exit the program
    if cv2.waitKey(21) & 0xFF == ord("q"):
        break
    
# Release the video capture device and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

# Close the CSV file
f.close()
