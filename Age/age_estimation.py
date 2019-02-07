from keras.models import model_from_json
import cv2
import sys
import numpy as np
from mtcnn.mtcnn import MTCNN


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 3
    # load json and create model
    json_file = open('ssrnet_3_3_3_64_1.0_1.0.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("ssrnet_3_3_3_64_1.0_1.0.h5")
    print("Loaded model from disk")
    img_size = 64
    # Load face Detector
    detector = MTCNN()
    imagePath = sys.argv[1]
    # Read the image
    image = cv2.imread(imagePath)
    # image = cv2.resize(image, (1280, 720))
    ad = 0.4
    input_img = image
    detected = detector.detect_faces(input_img)
    img_h, img_w, _ = np.shape(input_img)
    input_img = cv2.resize(input_img, (1024, int(1024 * img_h / img_w)))
    img_h, img_w, _ = np.shape(input_img)

    faces = np.empty((len(detected), img_size, img_size, 3))

    for i, d in enumerate(detected):
        if d['confidence'] > 0.95:
            x1, y1, w, h = d['box']
            x2 = x1 + w
            y2 = y1 + h
            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            try:
                faces[i, :, :, :] = cv2.resize(image[yw1:yw2, xw1:xw2, :], (img_size, img_size))
            except cv2.error:
                faces[i, :, :, :] = cv2.resize(image[yw1:yw2-10, xw1:xw2-10, :], (img_size, img_size))

    if len(detected) > 0:
        # predict ages and genders of the detected faces
        results = loaded_model.predict(faces)
        predicted_ages = results
        print(predicted_ages)
    print('Detect {} faces!'.format(len(detected)))
    # draw results
    for i, d in enumerate(detected):
        if d['confidence'] > 0.95:
            x1, y1, w, h = d['box']
            label = "{}".format(int(predicted_ages[i][0]))
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            x, y = (x1, y1)
            cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
            cv2.putText(image, label, (x1, y1), font, font_scale, (255, 255, 255), thickness)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()