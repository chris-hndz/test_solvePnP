import numpy as np
import cv2
import math

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return [x, y, z, w]

# Puntos 3D (sistema de coordenadas de Unity)
object_points = np.array([
    [11.00, 22.00, -5.00],
    [24.00, -13.00, -37.00],
    [-1.00, 15.00, -3.00],
    [-10.00, -9.00, -15.00],
    [5.00, 9.00, 3.00],
    [2.00, 7.00, 4.00]
], dtype=np.float32)

# Convertir puntos 3D al sistema de coordenadas de OpenCV
object_points_opencv = object_points.copy()
object_points_opencv[:, 1] *= -1  # Invertir Y
object_points_opencv[:, 2] *= -1  # Invertir Z

# Puntos 2D
imgHeight = 1920
image_points = np.array([
    [126.16, imgHeight - 1684.23],
    [193.30, imgHeight - 613.30],
    [778.76, imgHeight - 1377.82],
    [943.50, imgHeight - 432.35],
    [318.30, imgHeight - 1070.85],
    [669.33, imgHeight - 830.67]
], dtype=np.float32)

# Matriz de cámara ajustada
camera_matrix = np.array([
    [775.9587, 0, 540],
    [0, 775.9587, 960],
    [0, 0, 1]
], dtype=np.float32)

# Coeficientes de distorsión (asumiendo que no hay distorsión)
dist_coeffs = np.zeros((4, 1))

# Resolver PnP
success, rvec, tvec = cv2.solvePnP(object_points_opencv, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

# Convertir rvec a matriz de rotación
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Ajustar la matriz de rotación para el sistema de coordenadas de Unity
rotation_matrix_unity = np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]]) @ rotation_matrix.T

# Extraer ángulos de Euler de la matriz de rotación
sy = math.sqrt(rotation_matrix_unity[0,0] * rotation_matrix_unity[0,0] +  rotation_matrix_unity[1,0] * rotation_matrix_unity[1,0])
singular = sy < 1e-6

if not singular:
    x = math.atan2(rotation_matrix_unity[2,1], rotation_matrix_unity[2,2])
    y = math.atan2(-rotation_matrix_unity[2,0], sy)
    z = math.atan2(rotation_matrix_unity[1,0], rotation_matrix_unity[0,0])
else:
    x = math.atan2(-rotation_matrix_unity[1,2], rotation_matrix_unity[1,1])
    y = math.atan2(-rotation_matrix_unity[2,0], sy)
    z = 0

# Convertir a grados
euler_angles_unity = np.degrees([x, y, z])

# Ajustar los ángulos para el sistema de Unity
euler_angles_unity[0] = -euler_angles_unity[0]
euler_angles_unity[2] = -euler_angles_unity[2]

# Asegurarse de que los ángulos estén en el rango [-180, 180]
euler_angles_unity = [(angle + 180) % 360 - 180 for angle in euler_angles_unity]

# Ajustar el ángulo Y para que sea exactamente -180 si está muy cerca
if abs(euler_angles_unity[1] + 180) < 1e-5:
    euler_angles_unity[1] = -180

# Convertir ángulos de Euler a quaternion
quaternion = euler_to_quaternion(np.radians(euler_angles_unity[0]), np.radians(euler_angles_unity[1]), np.radians(euler_angles_unity[2]))

# Ajustar el vector de traslación para el sistema de coordenadas de Unity
tvec_unity = np.array([-tvec[0], tvec[1], tvec[2]])

# Redondear los resultados para una mejor presentación
tvec_unity = np.round(tvec_unity, decimals=8)
euler_angles_unity = np.round(euler_angles_unity, decimals=8)
quaternion = np.round(quaternion, decimals=8)

# Imprimir resultados
print("Posición de la cámara (Unity):\n", tvec_unity)
print("Rotación de la cámara (Unity Euler angles):\n", euler_angles_unity)
print("Rotación de la cámara (Quaternion):\n", quaternion)

# Proyectar puntos 3D a 2D y calcular el error de reproyección
projected_points, _ = cv2.projectPoints(object_points_opencv, rvec, tvec, camera_matrix, dist_coeffs)
error = cv2.norm(image_points, projected_points.reshape(-1, 2), cv2.NORM_L2) / len(object_points)
print(f"Error de reproyección promedio: {error} píxeles")