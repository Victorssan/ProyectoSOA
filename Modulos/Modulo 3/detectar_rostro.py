import cv2 as cv
import conexion_camara_pb2  # Importa las definiciones de mensajes de tu archivo .proto

def detectar_rostros(imagen):
    # Carga el clasificador preentrenado para la detección de rostros
    clasificador = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convierte la imagen a escala de grises
    imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    # Realiza la detección de rostros
    rostros = clasificador.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibuja un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in rostros:
        cv.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return imagen

def main():
    # Abre la conexión con la cámara
    cam = cv.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        # Llama a la función de detección de rostros
        frame_con_rostros = detectar_rostros(frame)

        # Muestra el fotograma con los rostros detectados
        cv.imshow('Detección de Rostros', frame_con_rostros)

        # Si se presiona 'q', sal del bucle
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos de la cámara y cierra las ventanas
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
