import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import vlc
import sys
import mensaje_main_pb2

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (2 * euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
        shape[33],
        shape[8],
        shape[45],
        shape[36],
        shape[54],
        shape[48]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return translation_vector[1][0]

def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

def ear(eye):
    return ((euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3])))

def writeEyes(a, b, img):
    y1_left = max(a[1][1], a[2][1])
    y2_left = min(a[4][1], a[5][1])
    x1_left = a[0][0]
    x2_left = a[3][0]

    cv2.imwrite('left-eye.jpg', img[y1_left:y2_left, x1_left:x2_left])

    y1_right = max(b[1][1], b[2][1])
    y2_right = min(b[4][1], b[5][1])
    x1_right = b[0][0]
    x2_right = b[3][0]

    cv2.imwrite('right-eye.jpg', img[y1_right:y2_right, x1_right:x2_right])

alert = vlc.MediaPlayer('alert-sound.mp3')

frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5
close_thresh = 0.3
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

capture = cv2.VideoCapture(0)

avgEAR = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Inicializar eCAL
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher

ecal_core.initialize(sys.argv, "Python Somnolencia Detector Publisher")
pub = ProtoPublisher("somnolencia_data_YISUS", mensaje_main_pb2.Somnolencia)


while(True):
    ret, frame = capture.read()
    size = frame.shape
    gray = frame
    rects = detector(gray, 0)

    if len(rects) > 0:
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEAR = ear(leftEye)
        rightEAR = ear(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0
        eyeContourColor = (255, 255, 255)

        if yawn(shape[mStart:mEnd]) > 0.6:
            cv2.putText(gray, "Bostezo Detectado", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            yawn_countdown = 1

        if avgEAR < close_thresh:
            flag += 1

            if yawn_countdown and flag >= frame_thresh_3:
                cv2.putText(gray, "Somnolencia despues de bostezo", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                alert.play()
                if map_flag:
                    map_flag = 0
                    map_counter += 1
            elif flag >= frame_thresh_2 and getFaceDirection(shape, size) < 0:
                cv2.putText(gray, "Somnolencia (Postura del Cuerpo)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                alert.play()
                if map_flag:
                    map_flag = 0
                    map_counter += 1
            elif flag >= frame_thresh_1:
                cv2.putText(gray, "Somnolencia (Normal)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                alert.play()
                if map_flag:
                    map_flag = 0
                    map_counter += 1

        elif avgEAR > close_thresh and flag:
            print("Contador restablecido a 0")
            alert.stop()
            yawn_countdown = 0
            map_flag = 1
            flag = 0

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)

    if avgEAR > close_thresh:
        alert.stop()

    cv2.imshow('Conductor', gray)

    if cv2.waitKey(1) == 27:
        break

    # Publicar mensaje eCAL
    protobuf_message = mensaje_main_pb2.Somnolencia()
    protobuf_message.avg_ear = avgEAR
    pub.send(protobuf_message)

# Liberar la captura de video y cerrar las ventanas
capture.release()
cv2.destroyAllWindows()
ecal_core.finalize()