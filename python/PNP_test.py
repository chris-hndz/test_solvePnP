import numpy as np
import cv2
import math
#Puntos 3D 
object_points = np.array([[7.00, 13.00, -5.00],[20.00, -22.00, -37.00],[-5.00, 6.00, -3.00],[-11.00, -18.00, -15.00],[1.00, 0.00, 3.00]], dtype=np.float32)#np.array([[ -0.17, 0.30,  5.79],[0.73, 0.30, 5.36],[ 0.58, -0.64, 5.05],[ -0.32,  -0.64, 5.49],[-0.58,  0.64, 4.95],[0.32, 0.64, 4.51],[ 0.17, -0.30,  4.21],[-0.73,  -0.30,  4.64]], dtype=np.float32)
image_points = np.array([[177.89, 1632.50],[209.80, 596.79],[838.45, 1318.14],[881.42, 401.31],[429.15, 960.00]], dtype=np.float32)#np.array([[601.19, 1067.54],[304.46, 1057.10],[363.95, 765.79],[646.38, 747.43],[712.30, 1150.07],[452.57, 1134.7],[495.57, 881.92],[744.03, 875.89]], dtype=np.float32)

camera_matrix = np.array([
    [775.9587, 0, 448],  # Focal length x, skew, center x
    [0, 775.9587, 448],  # Skew, focal length y, center y
    [0, 0, 1]        
], dtype=np.float32)

#Sin distorsion
dist_coeffs = np.zeros((4, 1)) 

success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
rotation_matrix, _ = cv2.Rodrigues(rvec)
projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)


print("Vector de rotacion (rvec):\n", rvec*180/math.pi)
print("Vector de traslacion (tvec):\n", tvec)
print("Matriz de rotacion:\n", rotation_matrix)
print("Puntos proyectados:\n", projected_points)