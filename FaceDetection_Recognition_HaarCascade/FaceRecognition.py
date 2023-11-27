import cv2                                                                                                  # Import OpenCV library
import os                                                                                                   # Import os module for operating system functions
import matplotlib.pyplot as plt                                                                             # Import matplotlib for plotting

def preprocess_image(image_path):
    img = cv2.imread(image_path)                                                                            # Read the image
    img = cv2.resize(img, (800, 600))                                                                       # Resize the image for better processing speed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                            # Convert the image to grayscale
    gray = cv2.equalizeHist(gray)                                                                           # Apply histogram equalization for better contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)                                                                # Apply Gaussian blur to reduce noise and improve face detection
    return gray

def preprocess_video(frame):
    frame = cv2.resize(frame, (800, 600))                                                                   # Resize the video frame for better processing speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                          # Convert the frame to grayscale
    gray = cv2.equalizeHist(gray)                                                                           # Apply histogram equalization for better contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)                                                                # Apply Gaussian blur to reduce noise and improve face detection
    return gray

def detect_faces_in_image(image_path, output_dir='output', output_filename='output_image.jpg'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')     # Load the pre-trained Haar Cascade model for face detection
    gray = preprocess_image(image_path)                                                                     # Pre-process the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)                            # Detect faces in the pre-processed image
    img = cv2.imread(image_path)                                                                            # Draw rectangles around the faces on the original image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    output_path = os.path.join(output_dir, output_filename)                                                 # Save the output image to the specified output directory
    cv2.imwrite(output_path, img)
    print(f"Face detection complete. Result saved to {output_path}")

    num_detected_faces = len(faces)                                                                         # Count the number of faces detected

    return num_detected_faces

def detect_faces_in_video(video_path, output_dir='output', output_filename='output_video.avi'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')     # Load the pre-trained Haar Cascade model for face detection
    cap = cv2.VideoCapture(video_path)                                                                      # Open the video file
    width = int(cap.get(3))                                                                                 # Get the video width
    height = int(cap.get(4))                                                                                # Get the video height

    fourcc = cv2.VideoWriter_fourcc(*'XVID')                                                                # Save the output video to the specified output directory
    output_path = os.path.join(output_dir, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = preprocess_video(frame)                                                                      # Pre-process the video frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)                        # Detect faces in the pre-processed video frame

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)                                        # Draw rectangles around the faces on the original frame

        out.write(frame)                                                                                    # Write the frame to the output video

    cap.release()                                                                                           # Release the VideoCapture object
    out.release()                                                                                           # Release the VideoWriter object

    print(f"Face detection complete. Result saved to {output_path}")

    num_detected_faces = len(faces)                                                                         # Count the number of faces detected

    return num_detected_faces

if __name__ == "__main__":
    folder_path = 'D:\Github\FaceRocgnition\FolderForImageAndVideoFIles'                                    # Specify the path to the folder containing images and videos
    output_folder_path = 'D:\Github\FaceRocgnition\output'                                                  # Specify the output directory

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]             # Process images in the folder
    actual_face_counts_image = []
    detected_face_counts_image = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        actual_count = int(input(f"Enter the actual number of faces in {image_file}: "))                    # Ask the user for the actual count of faces
        detected_count = detect_faces_in_image(image_path, output_dir=output_folder_path, output_filename='output_' + image_file)  # Detect faces in the image and get the detected count
        actual_face_counts_image.append(actual_count)
        detected_face_counts_image.append(detected_count)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]                      # Process videos in the folder
    frame_counts = []
    detected_face_counts_video = []

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        actual_counts_per_frame = [int(x) for x in input(f"Enter the actual number of faces for each frame in {video_file} (comma-separated): ").split(',')]  # Ask the user for the actual count of faces per frame
        detected_count = detect_faces_in_video(video_path, output_dir=output_folder_path, output_filename='output_' + video_file)  # Detect faces in the video and get the detected count
        frame_counts.extend(actual_counts_per_frame)
        detected_face_counts_video.extend([detected_count] * len(actual_counts_per_frame))

    plt.figure(figsize=(12, 6))                                                                             # Plotting for images                                                                                                            
    plt.subplot(1, 2, 1)
    plt.plot(image_files, actual_face_counts_image, marker='o', label='Actual Count', color='blue')
    plt.plot(image_files, detected_face_counts_image, marker='o', label='Detected Count', color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.title('Image Face Detection Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)                                                                                    # Plotting for videos
    plt.scatter(range(1, len(frame_counts) + 1), frame_counts, label='Actual Count per Frame', color='blue')
    plt.scatter(range(1, len(detected_face_counts_video) + 1), detected_face_counts_video, label='Detected Count', color='orange')
    plt.title('Video Face Detection Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()
