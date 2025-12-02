# Face Swapping Tool

A desktop application that swaps faces between two photos using computer vision and MediaPipe face detection.

## What it does

<img width="1915" height="992" alt="image" src="https://github.com/user-attachments/assets/fd0deca2-d624-43c1-9bbc-ece6e2be024a" />


This tool lets you upload two images with faces and automatically swaps them. You'll see a side-by-side comparison showing:
- Your original images at the top
- The swapped results at the bottom

The app uses MediaPipe's face mesh to detect 478 facial landmarks, which makes the swap pretty accurate. It blends the faces naturally using OpenCV's seamless cloning technique.

## Features

- Easy to use GUI with drag-and-drop image selection
- Automatic face detection and landmark mapping
- Side-by-side comparison view (2x2 grid)
- Save the final result as an image file
- Works with most common image formats (PNG, JPG, etc.)

## How to use

Install the dependencies:
```bash
pip install opencv-python mediapipe scipy pillow
```

Run the program:
```bash
python face_swapp.py
```

Then just:
1. Click "Select Image 1" and choose the first photo
2. Click "Select Image 2" and choose the second photo
3. Click "Swap Faces" to see the magic happen
4. Use "Save Image" to export your result

## Requirements

- Python 3.7 or higher
- A computer with a decent CPU (MediaPipe can be a bit heavy)
- Images with clearly visible faces work best

## Notes

The face detection works best when:
- Faces are looking forward or at slight angles
- There's good lighting in the photos
- The face isn't covered by hands, masks, or other objects

If the swap doesn't work, try using different photos with clearer face visibility.
