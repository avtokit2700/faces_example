import cv2
import sys
import dlib

# Get user supplied values
imagePath = sys.argv[1]
hog_face_detector = dlib.get_frontal_face_detector()
# Read the image
image = cv2.imread(imagePath)
image = cv2.resize(image, (1280, 720))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces_hog = hog_face_detector(image, 1)

# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow("face detection with dlib", image)
cv2.waitKey()