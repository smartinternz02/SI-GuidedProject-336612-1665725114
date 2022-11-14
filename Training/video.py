#Video streaming
import cv2
#import facevec
import numpy as np
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models  import load_model

model = load_model(r'C:\Users\sripa\alert.h5') 
video = cv2.VideoCapture(0)
cv2.namedWindow("Window")
name = ["Human","Wild aniaml","otimher"]
    
while(1):
    success, frame = video.read()
    cv2.imwrite("image.jpg",frame)
    img = image.load_img("image.jpg",target_size = (64,64))
    x  = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    pred = model.predict(x)
    p = int(pred[0][0])
    print(pred)
    cv2.putText(frame, "predicted  class = "+str(name[p]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

video.release()
cv2.destroyAllWindows()
