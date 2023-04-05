import cv2

# allumer la caméra
cap = cv2.VideoCapture(0)

# charger le fichier XML contenant les informations sur les visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # capturer une image depuis la caméra
    ret, frame = cap.read()

    # convertir l'image en noir et blanc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # dessiner un rectangle autour de chaque visage détecté
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # afficher l'image avec les rectangles
    cv2.imshow('frame',frame)

    # attendre une touche pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
