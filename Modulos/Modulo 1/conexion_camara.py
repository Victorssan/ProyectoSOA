# conexion_camara.py:
# establecemos una conexión con una cámara y envíamos información sobre esa conexión
import sys
import cv2 as cv
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import conexion_camara_pb2

def establecer_conexion_camara():
    # Inicializamos eCAL
    ecal_core.initialize(sys.argv, "Python Conexion con Camara")

    # Creamos un publicador para enviar información sobre la conexión
    pub = ProtoPublisher("conexion_camara", conexion_camara_pb2.ConexionCamara)

    # Configuramos la cámara, 0 es la cámara predeterminada
    cam = cv.VideoCapture(0)

    # Verificamos si la cámara se pudo abrir correctamente
    if not cam.isOpened():
        print("No se pudo conectar a la cámara.")

    else:
        # Obtenemos información de la cámara, como resolución y tasa de fotogramas
        ancho = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        alto = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        tasa_fotogramas = cam.get(cv.CAP_PROP_FPS)

        # Creamos un mensaje protobuf con la información de la cámara
        conexion_camara = conexion_camara_pb2.ConexionCamara()
        conexion_camara.nombre = "Camara1"
        conexion_camara.resolucion_ancho = ancho
        conexion_camara.resolucion_alto = alto
        conexion_camara.tasa_fotogramas = tasa_fotogramas

        # Enviamos el mensaje protobuf con la información de la conexión
        pub.send(conexion_camara)

        print("Conexión exitosa con la cámara.")

    # Finalizamos eCAL
    ecal_core.finalize()

if __name__ == "__main__":
    establecer_conexion_camara()
