import cv2
from mtcnn.mtcnn import MTCNN
import sys

# Get user supplied values
imagePath = sys.argv[1]
# Read the image
image = cv2.imread(imagePath)
image = cv2.resize(image, (1280, 720))
detector = MTCNN()
result = detector.detect_faces(image)
for i in range(len(result)):
	bounding_box = result[i]['box']
	keypoints = result[i]['keypoints']
	cv2.rectangle(image,
	              (bounding_box[0], bounding_box[1]),
	              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
	              (0,155,255),
	              2)

cv2.imshow("face detection with dlib", image)
cv2.waitKey()