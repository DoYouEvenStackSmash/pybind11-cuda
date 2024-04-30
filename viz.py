#!/usr/bin/python3
import cv2
import numpy as np


def scatterplot(image_shape, points, radius=3, color=(255, 255, 255)):
    # Create a blank image
    scatter_image = np.zeros(image_shape, dtype=np.uint8)
    # Draw circles for each point
    points[points > 4000] = 1
    # points = points / 100
    points[0:3] = 66.6
    scale = np.max(points)

    scale_factor = image_shape[0] / scale

    for i, point in enumerate(points):
        # print(point)
        cv2.circle(
            scatter_image,
            (i, image_shape[0] - int(point[0] * scale_factor)),
            radius,
            color,
            -1,
        )
    return scatter_image
