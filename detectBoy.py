import cv2

img = cv2.imread("boy.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# loading the classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# detecting the features of face and getting tuple of x,y,w,h of the face

# detectMultiscale(image , scalefactor , minNeighbors) 
# scalefactor --> means the detection window size gets smaller
# and smaller after every round of detection, by the
# value chosen for scaleFactor to increase precision.
# Increasing the scaleFactor, helps to increase the detection accuracy.
# Possible range between 1.1 to 1.9

# minNeighbors: Parameter specifying how many facial features that need to be present, to detect the face.


face = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0) , 2)

    roi = img[y : y+h , x : x+w]
    cv2.imwrite("face.png" , roi)

cv2.imshow("Image" , img)
cv2.waitKey(0)






