import cv2
import numpy as np
import mediapipe as mp
from tkinter import filedialog, messagebox, ttk
from tkinter import *
from PIL import Image, ImageTk
import os
import threading
from scipy.spatial import Delaunay

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Global variables
image_path_a = ""
image_path_b = ""
panelA = None
panelB = None
panelC = None
root = None
progress_bar = None
is_processing = False

def get_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for lm in landmarks.landmark:
        points.append([int(lm.x * w), int(lm.y * h)])
    
    return np.array(points)

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    u, s, vt = np.linalg.svd(points1.T @ points2)
    r = (u @ vt).T
    
    # Properly reshape the translation vector
    translation = c2.reshape(2, 1) - (s2 / s1) * r @ c1.reshape(2, 1)
    transform = np.vstack([np.hstack(((s2 / s1) * r, translation)), np.array([[0., 0., 1.]])])
    return transform

def warp_image(image, transform, shape):
    warped = cv2.warpAffine(image, transform[:2], (shape[1], shape[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return warped

def correct_colors(im1, im2, landmarks1):
    # Use face contour landmarks for blur calculation (MediaPipe indices)
    # Left eye: 33, Right eye: 263
    if len(landmarks1) > 263:
        blur_amount = 0.4 * np.linalg.norm(landmarks1[33] - landmarks1[263])
    else:
        blur_amount = 15.0
    
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    blur_amount = max(1, min(blur_amount, 31))  # Clamp between 1 and 31
    
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    return np.clip(im2.astype(np.float32) + im1.astype(np.float32) - im1_blur.astype(np.float32), 0, 255).astype(np.uint8)

def select_image(is_first_image=True):
    global panelA, panelB, image_path_a, image_path_b

    image_path = filedialog.askopenfilename()
    if len(image_path) > 0:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if is_first_image:
            image_path_a = image_path
            if panelA is None:
                panelA = Label(image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)
            else:
                panelA.configure(image=image)
                panelA.image = image
        else:
            image_path_b = image_path
            if panelB is None:
                panelB = Label(image=image)
                panelB.image = image
                panelB.pack(side="right", padx=10, pady=10)
            else:
                panelB.configure(image=image)
                panelB.image = image

def swap_faces_thread():
    global is_processing
    is_processing = True
    progress_bar.start(10)
    swap_faces()
    progress_bar.stop()
    is_processing = False

def swap_faces():
    global panelC

    if not image_path_a or not image_path_b:
        messagebox.showerror("Error", "Please select both images first.")
        return

    try:
        image_a = cv2.imread(image_path_a)
        image_b = cv2.imread(image_path_b)

        if image_a is None or image_b is None:
            messagebox.showerror("Error", "Failed to load images.")
            return

        landmarks1 = get_landmarks(image_a)
        landmarks2 = get_landmarks(image_b)

        if landmarks1 is None or landmarks2 is None:
            messagebox.showerror("Error", "Could not detect faces. Make sure faces are clearly visible in both images.")
            return

        print(f"Detected {len(landmarks1)} landmarks in image 1")
        print(f"Detected {len(landmarks2)} landmarks in image 2")

        # Create a 2x2 grid: original images on top, swapped faces on bottom
        # Top row: original image A | original image B
        # Bottom row: B's face on A | A's face on B
        
        # Swap face B onto image A
        output1 = swap_face(image_b, image_a, landmarks2, landmarks1)
        
        # Swap face A onto image B
        output2 = swap_face(image_a, image_b, landmarks1, landmarks2)

        # Resize all images to same height for consistent display
        target_height = 400
        
        def resize_with_aspect(img, target_h):
            h, w = img.shape[:2]
            new_w = int(w * target_h / h)
            return cv2.resize(img, (new_w, target_h))
        
        img_a_resized = resize_with_aspect(image_a, target_height)
        img_b_resized = resize_with_aspect(image_b, target_height)
        output1_resized = resize_with_aspect(output1, target_height)
        output2_resized = resize_with_aspect(output2, target_height)
        
        # Add labels to images
        def add_label(img, text):
            labeled = img.copy()
            cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            return labeled
        
        img_a_labeled = add_label(img_a_resized, "Original A")
        img_b_labeled = add_label(img_b_resized, "Original B")
        output1_labeled = add_label(output1_resized, "B face on A")
        output2_labeled = add_label(output2_resized, "A face on B")
        
        # Create top row (originals)
        top_row = np.hstack((img_a_labeled, img_b_labeled))
        
        # Create bottom row (swapped)
        bottom_row = np.hstack((output1_labeled, output2_labeled))
        
        # Stack top and bottom rows
        output_image = np.vstack((top_row, bottom_row))
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)
        output_image = ImageTk.PhotoImage(output_image)

        if panelC is None:
            panelC = Label(root)
            panelC.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        panelC.configure(image=output_image)
        panelC.image = output_image
        
        print("Face swap completed successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Face swap failed: {str(e)}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def swap_face(src_img, dst_img, src_landmarks, dst_landmarks):
    """Swap face from src_img onto dst_img"""
    
    # Create output image
    output = dst_img.copy()
    
    # Get convex hull of destination face
    hull = cv2.convexHull(dst_landmarks)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(hull)
    center = (x + w // 2, y + h // 2)
    
    # Calculate transformation
    transform = transformation_from_points(dst_landmarks, src_landmarks)
    
    # Warp source face to destination
    warped = warp_image(src_img, transform, dst_img.shape)
    
    # Create mask
    mask = np.zeros(dst_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Blend using seamlessClone
    try:
        # Make sure center is within bounds
        h_img, w_img = dst_img.shape[:2]
        center = (max(w//2, min(center[0], w_img - w//2)), 
                  max(h//2, min(center[1], h_img - h//2)))
        
        output = cv2.seamlessClone(warped, dst_img, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print(f"SeamlessClone failed: {e}, using direct blend")
        # Fallback: direct blending
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (warped * mask_3ch + dst_img * (1 - mask_3ch)).astype(np.uint8)
    
    return output

def save_image():
    if panelC is None or not hasattr(panelC, 'image'):
        messagebox.showerror("Error", "No image to save. Please swap faces first.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
    )
    if file_path:
        try:
            # Get the image from the PhotoImage
            image = ImageTk.getimage(panelC.image)
            image.save(file_path)
            messagebox.showinfo("Success", f"Image saved successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def create_gui():
    global root, progress_bar

    root = Tk()
    root.title("Face Swapper")

    # Create main frame
    main_frame = Frame(root)
    main_frame.pack(padx=10, pady=10)

    # Create buttons frame
    buttons_frame = Frame(main_frame)
    buttons_frame.pack(side="top", fill="x")

    btnSelect1 = Button(buttons_frame, text="Select Image 1", command=lambda: select_image(True))
    btnSelect1.pack(side="left", padx=5)

    btnSelect2 = Button(buttons_frame, text="Select Image 2", command=lambda: select_image(False))
    btnSelect2.pack(side="left", padx=5)

    btnSwap = Button(buttons_frame, text="Swap Faces", command=lambda: threading.Thread(target=swap_faces_thread).start())
    btnSwap.pack(side="left", padx=5)

    # Create save button
    btnSave = Button(main_frame, text="Save Image", command=save_image)
    btnSave.pack(side="top", fill="x", pady=5)

    # Create progress bar
    progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=200, mode="indeterminate")
    progress_bar.pack(side="top", fill="x", pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()