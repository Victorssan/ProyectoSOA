# captura_video.py:
# utilizaremos una cámara web en el dispositivo para tomar fotos periódicamente.
import sys
import time
import cv2 as cv
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import conexion_camara_pb2

def establecer_conexion_camara_y_capturar():
    # Inicializamos eCAL
    ecal_core.initialize(sys.argv, "Python Conexion con Camara")

    # Creamos un publicador para enviar información de la conexión
    pub = ProtoPublisher("conexion_camara", conexion_camara_pb2.ConexionCamara)

    # Configuramos la cámara, 0 es la cámara predeterminada
    cam = cv.VideoCapture(0)

    # Obtención de información de la cámara
    ancho = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    alto = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    tasa_fotogramas = cam.get(cv.CAP_PROP_FPS)

    # Creamos un mensaje protobuf con información de la cámara
    conexion_camara = conexion_camara_pb2.ConexionCamara()
    conexion_camara.nombre = "Camara1"
    conexion_camara.resolucion_ancho = ancho
    conexion_camara.resolucion_alto = alto
    conexion_camara.tasa_fotogramas = tasa_fotogramas

    # Enviamos el mensaje protobuf con la información de la conexión
    pub.send(conexion_camara)

    # Iniciamos a capturar video periodicamente
    while True:
        ret_val, img = cam.read()
        if ret_val:
            # Mostramos la imagen
            cv.imshow('Captura de Video', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberamos los recursos de la cámara
    cam.release()

    # Finalizamos eCAL
    ecal_core.finalize()

if __name__ == "__main__":
    establecer_conexion_camara_y_capturar()
