import cv2

# Callback function to get the coordinates on mouse click
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left button of the mouse is clicked
        print(f'Coordinates: (X: {x}, Y: {y})')

# Capture the video
cap = cv2.VideoCapture('.\\testvideos\\230824_jeongseok.mp4')  # Replace with your video path

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('Video')  # Create a named window
cv2.setMouseCallback('Video', click_event)  # Bind the click_event function to the mouse click

# Read and display frame by frame
while True:
    ret, img = cap.read()  # Read a new frame
    original_width = img.shape[1]
    original_height = img.shape[0]
    # 크기 (큼)

    new_width = 1500
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    img = cv2.resize(img, (new_width, new_height)) #작게 만듦
    if not ret:
        print("Error: No frame captured from the video. Exiting...")
        break

    cv2.imshow('Video', img)  # Display the frame

    key = cv2.waitKey(20)  # Wait for 20 ms
    if key == 27:  # If 'Esc' key is pressed, break the loop
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
