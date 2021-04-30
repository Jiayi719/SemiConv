import cv2
import numpy as np


# Create a dot image.
NUM_DOTS_PER_ROW = 5
RADIUS_PIXEL = 2
num_classes = NUM_DOTS_PER_ROW ** 2


def gen_synthetic_data():
    image_size = NUM_DOTS_PER_ROW * (RADIUS_PIXEL * 3) + RADIUS_PIXEL
    image = np.zeros((image_size, image_size, 3), np.float32)
    label = np.zeros((image_size, image_size), np.uint8)

    for i in range(NUM_DOTS_PER_ROW):
        for j in range(NUM_DOTS_PER_ROW):
            cv2.circle(image, (i * (RADIUS_PIXEL * 3) + 2 * RADIUS_PIXEL, j * (RADIUS_PIXEL * 3) + 2 * RADIUS_PIXEL),
                       RADIUS_PIXEL, (255, 255, 255), -1)
            cv2.circle(label, (i * (RADIUS_PIXEL * 3) + 2 * RADIUS_PIXEL, j * (RADIUS_PIXEL * 3) + 2 * RADIUS_PIXEL),
                       RADIUS_PIXEL, (i * NUM_DOTS_PER_ROW + j) + 1, -1)

    return image, label


if __name__ == "__main__":
    image, label = gen_synthetic_data()
    cv2.imshow("Image", image)
    cv2.imshow("Label", label)
    cv2.waitKey(0)
