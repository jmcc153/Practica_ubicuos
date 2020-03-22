import cv2
import numpy as np


cap = cv2.VideoCapture(0)


cap.set(3, 480) #fija el ancho del frame
cap.set(4, 640) #fija el alto del frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Hombre', 'Mujer']
def load_caffe_models():
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel') #archivos entrenados para reconociminto de genero
    return(gender_net)
def video_detector(gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, image = cap.read()
       
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') #funcion para identificar objetos rapidamnte, en este caso rostros
 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5) #detecta objetos de diferentes tamanos de una imagen que en este caso es de gray
        if(len(faces)>0): #devuelve el numero de objetos, que en este caso es el numero de rostros 
            print("Found {} faces".format(str(len(faces))))
            for (x, y, w, h )in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
        #obtiene el rostro 
                face_img = image[y:y+h, h:h+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  #funcion para hacer el procesamiento de la imagen con un entrenamiento ya obtenido (deep learning)
        #Prediccion del genero
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Gender : " + gender)
        
                overlay_text = "%s" % (gender)
                cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', image)  
#0xFF is a hexadecimal constant which is 11111111 in binary.
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
if __name__ == "__main__":
    gender_net = load_caffe_models()
    video_detector(gender_net)
    