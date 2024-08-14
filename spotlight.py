import os
import cv2
import dlib
import numpy as np
import mediapipe as mp
import face_recognition
from PIL import Image
from filters import *

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.3)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_face_data(image):
    """Get face bounding boxes, landmarks, and encodings from an image."""
    rects = detect_faces(image)
    if not rects:
        return [], [], []
    landmarks = get_landmarks_dlib(image, rects)
    encodings = get_face_encodings(image, rects)
    return rects, landmarks, encodings

def get_landmarks_dlib(image, rects):
    """Get facial landmarks using Dlib, given bounding boxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = []

    h, w, _ = image.shape

    for rect in rects:
        x_min = int(rect.xmin * w)
        y_min = int(rect.ymin * h)
        width = int(rect.width * w)
        height = int(rect.height * h)
        
        x_max = x_min + width
        y_max = y_min + height
        dlib_rect = dlib.rectangle(left=x_min, top=y_min, right=x_max, bottom=y_max)
        
        # Get landmarks
        shape = predictor(gray, dlib_rect)
        face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks.append(face_landmarks)
    return landmarks

def process_image(input_data, filter_type,reference_image_path=None):
    """Process an image: detect faces, landmarks, and perform recognition."""
    image = cv2.imread(input_data) if isinstance(input_data, str) else np.array(input_data)

    rects, landmarks, face_encodings = get_face_data(image)
    recognized_face_indices = set()

    if reference_image_path:
        reference_image = cv2.imread(reference_image_path)
        _, _, reference_encodings = get_face_data(reference_image)
        known_face_encoding = reference_encodings[0] if reference_encodings else None

        if known_face_encoding.any() is not None:
            for i, face_encoding in enumerate(face_encodings):
                if recognize_face(face_encoding, known_face_encoding):
                    recognized_face_indices.add(i)
                    
    for i in recognized_face_indices:
        image = apply_filter_to_face(image, landmarks[i], filter_type)  
    
    routes = get_faceline(landmarks)
    blurred_image = image.copy()
    for i, landmarks in enumerate(routes):
        if i not in recognized_face_indices:
            blurred_image = blur_paste([landmarks], blurred_image)
    return blurred_image

def apply_filter_to_face(img, face_landmarks, filter_type):
    """Apply custom filter or decoration to the target face."""
    face_loc = face_landmarks if face_landmarks.any() else None
    if not face_loc.any():
        print('Couldn''t detect the face landmark.')
        face_loc = default_face_landmarks(img)
    if filter_type == 'bigface':
        return enlarge_face(img, face_loc)
    elif filter_type == 'redline':
        return red_face(img, face_loc)
    elif filter_type == 'redArrow':
        return red_arrow(img,face_loc)
    elif filter_type == 'ImHere':
        return ImHere(img,face_loc)
    return img
    
def save_image(input_data, output_path, filter_type='redArrow',reference_image_path=None):
    """Process and save an image."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_image = process_image(input_data,filter_type,reference_image_path, )
    cv2.imwrite(output_path, output_image)
    print(f"Image saved to {output_path}")

def detect_faces(image):
    """Detect faces and return their bounding boxes using MediaPipe."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    if results.detections:
        # print(f"Detected {len(results.detections)} faces.")
        return [detection.location_data.relative_bounding_box for detection in results.detections]
    print("No faces detected.")
    return None

def get_face_encodings(image, rects):
    """Get face encodings for each detected face rectangle using face_recognition."""
    face_encodings = []
    h, w, _ = image.shape
    face_locations = []
    
    for rect in rects:
        x_min = int(rect.xmin * w)
        y_min = int(rect.ymin * h)
        width = int(rect.width * w)
        height = int(rect.height * h)
        
        x_max = x_min + width
        y_max = y_min + height
        
        # Ensure coordinates are within image bounds
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, w), min(y_max, h)
        face_location = (y_min, x_max, y_max, x_min)
        face_locations.append(face_location)
  
    encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    face_encodings.extend(encodings)
    
    return face_encodings

def recognize_face(face_encoding, known_face_encoding):
    """Compared detected face encoding with reference image."""
    return face_recognition.compare_faces([known_face_encoding], face_encoding)[0]

def get_faceline(landmarks):
    routes = []
    faces = []
    for p in range(len(landmarks)):
        for i in range(15, -1, -1):
            from_coordinate = landmarks[p][i+1]
            to_coordinate = landmarks[p][i]
            faces.append(from_coordinate)
        from_coordinate = landmarks[p][0]
        to_coordinate = landmarks[p][17]
        faces.append(from_coordinate)
        
        for i in range(17, 20):
            from_coordinate = landmarks[p][i]
            to_coordinate = landmarks[p][i+1]
            faces.append(from_coordinate)
        
        from_coordinate = landmarks[p][19]
        to_coordinate = landmarks[p][24]
        faces.append(from_coordinate)
        
        for i in range(24, 26):
            from_coordinate = landmarks[p][i]
            to_coordinate = landmarks[p][i+1]
            faces.append(from_coordinate)
        
        from_coordinate = landmarks[p][26]
        to_coordinate = landmarks[p][16]
        faces.append(from_coordinate)
        faces.append(to_coordinate)
        routes.append(faces)
        faces = []
    return routes

def blur_paste(routes, img): 
    mask = np.zeros_like(img) 
    for landmarks in routes:
        mask = cv2.fillConvexPoly(mask, np.array(landmarks), (255, 255, 255))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
    blurred_region = cv2.GaussianBlur(img, (61,61), 21)

    result_img = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    result_img = cv2.bitwise_or(result_img, cv2.bitwise_and(blurred_region, mask))
    return result_img
