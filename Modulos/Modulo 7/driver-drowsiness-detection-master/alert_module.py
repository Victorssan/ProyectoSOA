import vlc
import cv2
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
#import mensaje_main_pb2
import mensaje_main_pb2
from detector_module import close_thresh

def alert_sound():
    alert = vlc.MediaPlayer('alert-sound.mp3')
    alert.play()

def stop_alert(alert_instance):
    alert_instance.stop()

def main():
    # Inicializar eCAL
    ecal_core.initialize([], "Python Somnolencia Alert Subscriber")
    sub = ProtoSubscriber("somnolencia_data", mensaje_main_pb2.Somnolencia)

    alert_instance = vlc.MediaPlayer('alert-sound.mp3')

    while True:
        # Recibir mensaje eCAL
        protobuf_message = sub.receive()
        avgEAR = protobuf_message.avgEAR

        # Lógica de alerta
        if avgEAR < close_thresh:
            if not alert_instance.is_playing():
                alert_sound()
        else:
            # Detener la alerta si no se cumple la condición
            stop_alert(alert_instance)

        # Agregar alguna condición para salir del bucle, por ejemplo, presionar 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Finalizar eCAL y detener la alerta al salir del bucle
    ecal_core.finalize()
    stop_alert(alert_instance)

if __name__ == "__main__":
    main()
