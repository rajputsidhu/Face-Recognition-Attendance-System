import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox

# Initialize global variables
video_capture = None
f = None
known_face_encodings = []
known_face_names = []
students = []
now = None
lnwriter = None
current_subject = ""

# Function to start the face recognition process
def start_face_recognition(subject):
    global video_capture, f, known_face_encodings, known_face_names, students, now, lnwriter, current_subject

    current_subject = subject
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    try:
        load_known_faces()
    except (IndexError, FileNotFoundError) as e:
        messagebox.showerror("Error", str(e))
        return

    now = datetime.now()

    file_name = subject + '.csv'  # Save as subject name only

    try:
        f = open(file_name, 'a+', newline='')
        lnwriter = csv.writer(f)
    except Exception as e:
        messagebox.showerror("Error", f"Could not open CSV file. {e}")
        return

    process_frame()

# Function to load known faces
def load_known_faces():
    global known_face_encodings, known_face_names, students
    base_image_path = "/Users/sidhusingh/Devloper/Code/LEARNING PROJECT/PHOTOS/"
    
    images = ["Roshan 23cse252.jpg", "mukesh 23cse078.jpg", "malay 23cse032.png",
              "Anshuman 23cse074.jpg", "Aditya Raj 23cse003.jpg", "Sourav 23cse041.jpg"]
    names = ["Roshan23cse252", "Mukesh23cse078", "Malay23cse032",
             "Anshuman23cse074", "Aditya Raj23cse003", "Sourav23cse041"]

    known_face_encodings.clear()
    known_face_names.clear()

    for image, name in zip(images, names):
        image_path = os.path.join(base_image_path, image)

        if not os.path.exists(image_path):
            print(f"❌ Skipping {image} (File not found)")
            continue

        loaded_image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(loaded_image)

        if len(encodings) == 0:
            print(f"⚠ No face detected in {image}. Skipping...")
            continue

        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

    students = known_face_names.copy()
    print(f"✅ Loaded {len(known_face_encodings)} known faces.")

# Function to process video frames for face recognition
def process_frame():
    global video_capture, now

    ret, frame = video_capture.read()
    if not ret:
        messagebox.showerror("Error", "Could not read frame from camera.")
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    if not face_locations:
        root.after(10, process_frame)
        return

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.45:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        face_names.append(name)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])
            print(f"✅ {name} marked present at {current_time}")

    # Draw rectangles and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Show subject name on window
    cv2.putText(frame, f"Subject: {current_subject}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 255), 2)

    cv2.imshow('Face Recognition Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_face_recognition()
        return

    root.after(10, process_frame)

# Function to stop face recognition and release resources
def stop_face_recognition():
    global video_capture, f

    if video_capture is not None:
        video_capture.release()

    cv2.destroyAllWindows()

    if f is not None:
        f.close()

    messagebox.showinfo("Info", "Face recognition stopped and attendance saved.")

# Function to handle window closing event
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stop_face_recognition()
        root.destroy()

# === Tkinter GUI Setup ===
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x300")
root.protocol("WM_DELETE_WINDOW", on_closing)

selected_subject = tk.StringVar()
selected_subject.set("Select Subject")

# Subject Dropdown
tk.Label(root, text="Choose Subject:", font=("Arial", 14)).pack(pady=10)
subjects = ["DAA", "FPP", "ISC", "COA", "ACSPE"]
dropdown = tk.OptionMenu(root, selected_subject, *subjects)
dropdown.config(width=20, font=("Arial", 12))
dropdown.pack(pady=5)

# Start Button
def start_btn_clicked():
    subject = selected_subject.get()
    if subject == "Select Subject":
        messagebox.showwarning("Warning", "Please select a subject before starting.")
        return
    start_face_recognition(subject)
    start_button.config(state="disabled")
    stop_button.config(state="normal")

start_button = tk.Button(root, text="Start Attendance", font=("Arial", 14), command=start_btn_clicked)
start_button.pack(pady=10)

# Stop Button
def stop_btn_clicked():
    stop_face_recognition()
    start_button.config(state="normal")
    stop_button.config(state="disabled")

stop_button = tk.Button(root, text="Stop Attendance", font=("Arial", 14), command=stop_btn_clicked, state="disabled")
stop_button.pack(pady=10)

# Run GUI loop
root.mainloop()