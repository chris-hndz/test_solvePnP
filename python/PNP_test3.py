import numpy as np
import cv2
import math

object_points = np.array([
    [7.00, 13.00, -5.00],
    [20.00, -22.00, -37.00],
    [-5.00, 6.00, -3.00],
    [-11.00, -18.00, -15.00],
    [1.00, 0.00, 3.00],
    [-2.00, -2.00, 4.00]
], dtype=np.float32)

image_points = np.array([
    [177.89, 1632.50],
    [209.80, 596.79],
    [838.45, 1318.14],
    [881.42, 401.31],
    [429.15, 960.00],
    [798.65, 701.35]
], dtype=np.float32)

camera_matrix = np.array([
    [775.9587, 0, 448],
    [0, 775.9587, 448],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))

success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

rotation_matrix, _ = cv2.Rodrigues(rvec)
euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]

print("Vector de rotación (rvec) en grados:\n", rvec * 180 / math.pi)
print("Vector de traslación (tvec):\n", tvec)
print("Matriz de rotación:\n", rotation_matrix)
print("Ángulos de Euler (grados):", euler_angles)

projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
error = cv2.norm(image_points, projected_points.reshape(-1, 2), cv2.NORM_L2) / len(object_points)
print(f"Error de reproyección promedio: {error} píxeles")