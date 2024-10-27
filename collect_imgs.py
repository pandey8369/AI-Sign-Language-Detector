
import os
import cv2

def create_directories(base_dir, num_classes):
    """Create directories for each class if they do not exist."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for i in range(num_classes):
        class_dir = os.path.join(base_dir, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def test_camera(video_device):
    """Test if the video capture device can be opened."""
    cap = cv2.VideoCapture(video_device)
    
    if not cap.isOpened():
        print(f"Error: Could not open video device {video_device}.")
        return False
    
    print(f"Camera {video_device} successfully opened.")
    cap.release()
    return True

def capture_images(video_device, data_dir, num_classes, dataset_size):
    """Capture images from the video device and save them to the specified directory."""
    cap = cv2.VideoCapture(video_device)
    
    if not cap.isOpened():
        print(f"Error: Could not open video device {video_device}.")
        return
    
    for j in range(num_classes):
        class_dir = os.path.join(data_dir, str(j))
        print(f'Collecting data for class {j}')
        
        # Display a message to the user
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Error: Failed to capture initial frame.")
                continue
            
            cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break
        
        # Capture dataset_size number of images
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Error: Failed to capture image.")
                continue
            
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            img_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            counter += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATA_DIR = './data'
    NUMBER_OF_CLASSES = 10
    DATASET_SIZE = 100
    VIDEO_DEVICE = 0  # Change to 1, 2, etc., if needed

    create_directories(DATA_DIR, NUMBER_OF_CLASSES)
    
    # Test camera before starting image capture
    if not test_camera(VIDEO_DEVICE):
        print("Please check your camera connection and index.")
    else:
        capture_images(VIDEO_DEVICE, DATA_DIR, NUMBER_OF_CLASSES, DATASET_SIZE)
