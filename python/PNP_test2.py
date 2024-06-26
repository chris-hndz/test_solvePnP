import numpy as np
import cv2
import math

# Puntos 3D (invertir el eje Z para coincidir con el sistema de coordenadas de OpenCV)
object_points = np.array([
    [7.00, 13.00, 5.00],
    [20.00, -22.00, 37.00],
    [-5.00, 6.00, 3.00],
    [-11.00, -18.00, 15.00],
    [1.00, 0.00, -3.00],
    [-2.00, -2.00, -4.00]
], dtype=np.float32)

# Puntos 2D (invertir la coordenada Y para coincidir con el sistema de coordenadas de OpenCV)
imgHeight = 1920
image_points = np.array([
    [177.89, imgHeight - 1632.50],
    [209.80, imgHeight - 596.79],
    [838.45, imgHeight - 1318.14],
    [881.42, imgHeight - 401.31],
    [429.15, imgHeight - 960.00],
    [798.65, imgHeight - 701.35]
], dtype=np.float32)

# Matriz de cámara actualizada según CameraSetup.cs
camera_matrix = np.array([
    [775.9587, 0, 448],
    [0, 775.9587, 448],
    [0, 0, 1]
], dtype=np.float32)

# Sin distorsión
dist_coeffs = np.zeros((4, 1))

# Resolver PnP usando el método iterativo
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

# Convertir el vector de rotación a matriz de rotación
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Proyectar los puntos 3D usando los parámetros estimados
projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

# Imprimir resultados
print("Vector de traslación (tvec):\n", tvec.flatten())
print("Vector de rotación (rvec) en grados:\n", rvec.flatten() * 180 / math.pi)
print("Matriz de rotación:\n", rotation_matrix)
print("Puntos proyectados:\n", projected_points)

# Calcular el error de reproyección
error = cv2.norm(image_points, projected_points.reshape(-1, 2), cv2.NORM_L2) / len(object_points)
print(f"Error de reproyección promedio: {error} píxeles")

# Convertir la rotación a ángulos de Euler (en grados)
euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
print("Ángulos de Euler (grados):\n", euler_angles)