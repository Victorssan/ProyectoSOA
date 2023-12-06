# Importar bibliotecas necesarias
import cv2  # Biblioteca para trabajar con imágenes y videos
import math  # Biblioteca matemática para realizar operaciones matemáticas
import numpy as np  # Biblioteca para manipulación de datos en forma de arrays
import dlib  # Biblioteca para detectar rostros y puntos clave faciales
import imutils  # Funciones útiles para operaciones en imágenes
from imutils import face_utils  # Utilidades específicas para trabajar con caras en imágenes
from matplotlib import pyplot as plt  # Biblioteca para crear gráficos y visualizaciones
import vlc  # VLC para reproducir sonidos
import train as train  # Módulo de entrenamiento (no proporcionado en el código)

import sys, webbrowser, datetime  # Bibliotecas adicionales para funcionalidades específicas


# Función para calcular la apertura de la boca (bostezo)
def yawn(mouth):
    return (euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (
                2 * euclideanDist(mouth[0], mouth[6]))


# Función para calcular la dirección de la cara en términos de la coordenada y de la traslación 3D
def getFaceDirection(shape, size):
    # Coordenadas de puntos clave faciales relevantes para la cara y la boca
    image_points = np.array([
        shape[33],  # Puntero de la nariz
        shape[8],  # Barbilla
        shape[45],  # Esquina izquierda del ojo izquierdo
        shape[36],  # Esquina derecha del ojo derecho
        shape[54],  # Esquina izquierda de la boca
        shape[48]  # Esquina derecha de la boca
    ], dtype="double")

    # 3D puntos del modelo.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Puntero de la nariz
        (0.0, -330.0, -65.0),  # Barbilla
        (-225.0, 170.0, -135.0),  # Esquina izquierda del ojo izquierdo
        (225.0, 170.0, -135.0),  # Esquina derecha del ojo derecho
        (-150.0, -150.0, -125.0),  # Esquina izquierda de la boca
        (150.0, -150.0, -125.0)  # Esquina derecha de la boca
    ])

    # Configuración de la cámara para calcular la traslación y dirección 3D
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # No hay distorsión de lente
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return translation_vector[1][0]


# Función para calcular la distancia euclidiana entre dos puntos
def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# Función para calcular la relación de aspecto del ojo (EAR)
def ear(eye):
    return ((euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3])))


# Función para guardar imágenes de los ojos izquierdo y derecho
def writeEyes(a, b, img):
    # Coordenadas y delimitadores para extraer el ojo izquierdo
    y1_left = max(a[1][1], a[2][1])
    y2_left = min(a[4][1], a[5][1])
    x1_left = a[0][0]
    x2_left = a[3][0]

    # Guardar la región del ojo izquierdo como una imagen
    cv2.imwrite('left-eye.jpg', img[y1_left:y2_left, x1_left:x2_left])

    # Coordenadas y delimitadores para extraer el ojo derecho
    y1_right = max(b[1][1], b[2][1])
    y2_right = min(b[4][1], b[5][1])
    x1_right = b[0][0]
    x2_right = b[3][0]

    # Guardar la región del ojo derecho como una imagen
    cv2.imwrite('right-eye.jpg', img[y1_right:y2_right, x1_right:x2_right])


# Configuración de la alarma de sonido
alert = vlc.MediaPlayer('alert-sound.mp3')

# Umbrales para la detección de somnolencia
frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5
close_thresh = 0.3
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

# Inicialización de la captura de video
capture = cv2.VideoCapture(0)

# Variable para almacenar el promedio de la relación de aspecto del ojo (EAR)
#representar el valor promedio de la relación de aspecto del ojo
avgEAR = 0

# Crear un detector de caras frontal usando dlib
detector = dlib.get_frontal_face_detector()

# Cargar el modelo de predicción de puntos clave faciales (landmarks)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Obtener índices para los puntos clave faciales específicos
#asignar los índices de los puntos clave faciales relacionados con los ojos y la boca.
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Bucle principal del programa
while (True):
    # Leer un fotograma de la cámara
    ret, frame = capture.read()
    size = frame.shape
    gray = frame  # Convertir el fotograma a escala de grises (la detección facial funciona mejor en escala de grises)
    rects = detector(gray, 0)  # Detectar rostros en el fotograma
    if (len(rects)):
        # Detectar puntos clave faciales y extraer regiones de ojos y boca
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))  # Convertir los puntos clave a un formato NumPy
        leftEye = shape[leStart:leEnd]  # Extraer puntos clave del ojo izquierdo
        rightEye = shape[reStart:reEnd]  # Extraer puntos clave del ojo derecho
        leftEyeHull = cv2.convexHull(leftEye)  # Obtener el contorno convexo del ojo izquierdo
        rightEyeHull = cv2.convexHull(rightEye)  # Obtener el contorno convexo del ojo derecho
        leftEAR = ear(leftEye)  # Obtener la relación de aspecto del ojo izquierdo
        rightEAR = ear(rightEye)  # Obtener la relación de aspecto del ojo derecho
        avgEAR = (leftEAR + rightEAR) / 2.0  # Calcular el promedio de la relación de aspecto de ambos ojos
        eyeContourColor = (255, 255, 255)  # Color del contorno de los ojos por defecto (blanco)

        # Verificar si se detecta un bostezo y activar el contador
        if (yawn(shape[mStart:mEnd]) > 0.6):
            cv2.putText(gray, "Bostezo Detectado", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            yawn_countdown = 1

        # Verificar la somnolencia basada en la relación de aspecto del ojo
        if (avgEAR < close_thresh):
            flag += 1
            eyeContourColor = (0, 255, 255)
            print(flag)
            # Verificar si se cumplen las condiciones para la alarma y actualizar el contador del mapa
            if (yawn_countdown and flag >= frame_thresh_3):
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Somnolencia despues de bostezo", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 127), 2)
                alert.play()
                if (map_flag):
                    map_flag = 0
                    map_counter += 1
            elif (flag >= frame_thresh_2 and getFaceDirection(shape, size) < 0):
                eyeContourColor = (255, 0, 0)
                cv2.putText(gray, "Somnolencia (Postura del Cuerpo)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 127), 2)
                alert.play()
                if (map_flag):
                    map_flag = 0
                    map_counter += 1
            elif (flag >= frame_thresh_1):
                eyeContourColor = (0, 0, 255)
                cv2.putText(gray, "Somnolencia (Normal)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                alert.play()
                if (map_flag):
                    map_flag = 0
                    map_counter += 1
        # Si la somnolencia se corrige, restablecer las alarmas y contadores
        elif (avgEAR > close_thresh and flag):
            print("Contador restablecido a 0")
            alert.stop()
            yawn_countdown = 0
            map_flag = 1
            flag = 0

        # Dibujar contornos alrededor de los ojos
        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
        # Detener la alarma si la somnolencia se corrige
    if (avgEAR > close_thresh):
        alert.stop()
        # Mostrar la imagen en una ventana llamada 'Conductor'
    cv2.imshow('Conductor', gray)
    # Salir del bucle si se presiona la tecla 'Esc' (código ASCII 27)
    if (cv2.waitKey(1) == 27):
        break

# Liberar la captura de video y cerrar las ventanas
capture.release()
cv2.destroyAllWindows()
