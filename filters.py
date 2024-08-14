import cv2
import numpy as np

def default_face_landmarks(image, radius=50):
    """Create default face landmarks when the face landmark is not detected"""
    h, w, _ = image.shape
    center = (w // 2, h // 2)

    num_points = 68  # Number of face landmark points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [
        (
            center[0] + int(radius * np.cos(angle)),
            center[1] + int(radius * np.sin(angle))
        )
        for angle in angles
    ]
    return np.array(points)

def enlarge_face(img, face_landmarks):
    """Enlarge the face region based on the landmarks."""
    hull = cv2.convexHull(np.array(face_landmarks))
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))

    face_region = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(hull)
    #Adjust the size w*N, h*N if you want bigger face filter.
    enlarged_face = cv2.resize(face_region[y:y+h, x:x+w], (w*2, h*2)) 

    mask_resized = cv2.resize(mask[y:y+h, x:x+w], (w*2, h*2))
    mask_resized = mask_resized[:, :, 0]  
    
    # Calculate center position to paste enlarged face back
    center_x, center_y = x + w // 2, y + h // 2
    start_x, start_y = center_x - enlarged_face.shape[1] // 2, center_y - enlarged_face.shape[0] // 2

    # Place the enlarged face back into the original image
    result_img = img.copy()
    for i in range(0, enlarged_face.shape[0]):
        for j in range(0, enlarged_face.shape[1]):
            if mask_resized[i, j] > 0:  # Only copy non-background pixels
                if (0 <= start_y + i < result_img.shape[0]) and (0 <= start_x + j < result_img.shape[1]):
                    result_img[start_y + i, start_x + j] = enlarged_face[i, j]
    return result_img

def red_arrow(img, face_landmarks):
    """Place an arrow image below the lowest point of the face landmarks."""
    arrow_image_path = './images/filter/RedArrow.png'
    landmarks = np.array(face_landmarks, dtype=np.int32)
    lowest_point = landmarks[np.argmax(landmarks[:, 1])]
    arrow_img = cv2.imread(arrow_image_path, cv2.IMREAD_UNCHANGED)
    
    arrow_height, arrow_width = arrow_img.shape[:2]
    arrow_x = lowest_point[0] - arrow_img.shape[1] // 2
    arrow_y = lowest_point[1] + 10  
    arrow_x = max(0, min(arrow_x, img.shape[1] - arrow_img.shape[1]))
    arrow_y = max(0, min(arrow_y, img.shape[0] - arrow_img.shape[0]))
    roi_height = min(arrow_height, img.shape[0] - arrow_y)
    roi_width = min(arrow_width, img.shape[1] - arrow_x)
    
    # Get the region of interest (ROI) on the main image
    roi = img[arrow_y:arrow_y+roi_height, arrow_x:arrow_x+roi_width]

    # Overlay the arrow image onto the main image
    if arrow_img.shape[2] == 4:  # If arrow image has an alpha channel
        arrow_bgr = arrow_img[:, :, :3]
        arrow_alpha = arrow_img[:, :, 3] / 255.0
        
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - arrow_alpha) + arrow_bgr[:, :, c] * arrow_alpha
        
        img[arrow_y:arrow_y+roi_height, arrow_x:arrow_x+roi_width] = roi
    else:
        img[arrow_y:arrow_y+roi_height, arrow_x:arrow_x+roi_width] = arrow_img
    
    return img

def ImHere(img, face_landmarks):
    """Place an arrow image below the lowest point of the face landmarks."""
    arrow_image_path = './images/filter/ImHere.png'
    landmarks = np.array(face_landmarks, dtype=np.int32)
    lowest_point = landmarks[np.argmax(landmarks[:, 1])]
    arrow_img = cv2.imread(arrow_image_path, cv2.IMREAD_UNCHANGED)
    
    arrow_x = lowest_point[0] - arrow_img.shape[1] // 2
    arrow_y = lowest_point[1] + 10  

    # Ensure the arrow stays within image bounds
    arrow_x = max(0, min(arrow_x, img.shape[1] - arrow_img.shape[1]))
    arrow_y = max(0, min(arrow_y, img.shape[0] - arrow_img.shape[0]))
    
    if arrow_img.shape[2] == 4:  
        roi = img[arrow_y:arrow_y+arrow_img.shape[0], arrow_x:arrow_x+arrow_img.shape[1]]
        arrow_bgr = arrow_img[:, :, :3]
        arrow_alpha = arrow_img[:, :, 3] / 255.0
        
        # Blending
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - arrow_alpha) + arrow_bgr[:, :, c] * arrow_alpha
        img[arrow_y:arrow_y+arrow_img.shape[0], arrow_x:arrow_x+arrow_img.shape[1]] = roi
    else:
        img[arrow_y:arrow_y+arrow_img.shape[0], arrow_x:arrow_x+arrow_img.shape[1]] = arrow_img
    
    return img