import cv2

# Trained XML file for detecting faces
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
# Trained XML file for detecting smiles
smile_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_smile.xml')

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

    # Detect smile
    smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
    
    # To draw a rectangle in smile
    for (xx, yy, ww, hh) in smile:
        cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

# Display an image in a window 
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()