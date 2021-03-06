import cv2

# Trained XML file for detecting faces
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_eye.xml')

# read image files
img = cv2.imread('Image/person.jpg')

if img is None:
    print("Cannot read image file")
    exit()

# convert to gray scale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detects faces of different sizes in the input image 
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 2)

# detects how many faces are detected
print("Number of faces detected: ", len(faces))

for (x,y,w,h) in faces:
    # To draw a rectangle in a face 
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w] 
    roi_color = img[y:y+h, x:x+w] 
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray)  

    # To draw a rectangle in eyes 
    for (ex,ey,ew,eh) in eyes: 
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

# Display an image in a window     
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()