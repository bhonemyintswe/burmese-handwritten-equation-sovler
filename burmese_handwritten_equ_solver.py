import cv2 as cv
import numpy as np
from keras.models import load_model


def equ_solver(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rsz_img = cv.resize(gray_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

    gauss_blur_img = cv.GaussianBlur(rsz_img, (3, 3), cv.BORDER_CONSTANT)
    canny_img = cv.Canny(gauss_blur_img, 10, 150, L2gradient=True)

    contours, _ = cv.findContours(canny_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    model = load_model('burmese-hand-digit-math.h5')

    position_list = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        x, y, w, h = x - 20, y - 20, w + 40, h + 40
        if w >= 37 and h >= 37:
            position_list.append([x, y, w, h])
    position_list.sort()

    detected_imgs = []
    for (x, y, w, h) in position_list:

        new_img = gauss_blur_img[y:y + h, x:x + w]
        retval, thresh_crop = cv.threshold(new_img.copy(), thresh=200, maxval=255, type=cv.THRESH_BINARY_INV)

        detected_gray = cv.resize(thresh_crop, (28, 28), interpolation=cv.INTER_AREA)
        detected_imgs.append(detected_gray)

    detected_imgs_np = np.array(detected_imgs)
    detected_imgs_np = detected_imgs_np.reshape(detected_imgs_np.shape[0], 28, 28, 1).astype('float32')
    detected_imgs_np = detected_imgs_np / 255

    predictions = model.predict(detected_imgs_np)
    classes = []
    for i, prediction in enumerate(predictions):
        classes.append(np.argmax(prediction))

    return classes


