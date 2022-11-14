#Video streaming and alerting
#import opencv
import cv2
#import numpy
import numpy as np
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models  import load_model
#import Client from twilio API
from twilio.rest import Client
#import playsound package
from playsound import playsound

#Load saved model file using load_model method
model = load_model(r'C:\Users\sripa\alert.h5')
#To read webcam
video = cv2.VideoCapture(0)
#Type of classes or names of the labels that we considered
name = ['Human','Domestic', 'Wild']
#To execute the program repeatedly using while loop   
while(1):
    success, frame = video.read()
    cv2.imwrite("image.jpg",frame)
    img = image.load_img("image.jpg",target_size = (64,64))
    x  = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    pred = model.predict(x)
    p = int(pred[0][0])
    print(pred)
    cv2.putText(frame, "predicted  class = "+str(name[p]), (100,100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    
    pred = model.predict(x)
    if pred[0][0]==1:
        #twilio account ssid
        account_sid = 'ACbaedc0f433eb384b9fc9957a506c2b99'
        #twilo account authentication toke
        auth_token = '8a4eb2ee2f3ef2b3933896040415bd9e'
        client = Client(account_sid, auth_token)

        message = client.messages \
        .create(
         body='Danger!. Wild animal is detected, stay alert',
         from_='+15133275578', #the free number of twilio
         to='+919100588408')
        print(message.sid)
        print('Danger!!')
        print('Animal Detected')
        print ('SMS sent!')
        #playsound(r'C:\Users\DELL\Downloads\Tornado_Siren_II-Delilah-0.mp3')
        #break
    else:
        print("No Danger")
       #break
    cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

video.release()
cv2.destroyAllWindows()