from tkinter import Frame
from ast import arg
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import threading
from project import Capture
import string
import random
import DeepFace as dfe
from data import Data 
import json

global capture,rec_frame, grey, switch, neg, face, rec, out , cin ,cout, val, allFaces, timeInfo ,FaceId ,models ,backends, testVariable,output
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
cin = False
cout = False
val = 0
allFaces = []
timeInfo = {}
FaceId = None
models = Data.getModels()
output = {"Status":"",
          "Result":"",
          "Number":""}
backends = Data.getBackends()
#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
def checkout(img_name):
    global models,allFaces,timeInfo,models,backends,output
    output["Status"]="Face Captured, Checking Out..."
    entry_time = 0
    exit_time = 0
    try:
        fid = dfe.represent(img_path = "./{}".format(img_name),model_name = models[7],detector_backend = backends[2])
    except:
        output["Result"]="No Face Detected"
        print("No Face Detected")
        output["Number"]= "Number of People Checked In :- {}".format(len(allFaces))
        print("Number of People Checked In :- {}".format(len(allFaces)))
        return
    try:
        checkFace = dfe.find(img_representation = fid,representations = allFaces,model_name = models[7],detector_backend = backends[2])
    except:
        print("Error: Try again")
        output["Number"]= "Number of People Checked In :- {}".format(len(allFaces))
        print("Number of People Checked In :- {}".format(len(allFaces)))
        return
    if(checkFace.empty):
        output["Result"]="Face Not Found or Never Checked In"
        print("Face Not Found or Never Checked In")
    f = 0
    for i in checkFace['identity']:
        for j in allFaces:
            if(i==j[0]):
                entry_time = timeInfo[i]
                allFaces.remove(j)
                f=1
    exit_time = time.time();
    if(f==1):
        output["Result"]="CheckOut Complete"
        print("CheckOut Complete")
        print("Entry Time:- {}".format(entry_time))
        print("Exit_time :- {}".format(exit_time))
        print("Duration :- {}".format(exit_time-entry_time))
        output["Number"]= "Number of People Checked In :- {}".format(len(allFaces))
        print("Number of People Checked In :- {}".format(len(allFaces)))
def checkin(img_name):
    global faceId,allFaces,timeInfo,output

    output["Status"]="Face Captured, Detecting..."
    print("Face Captured, Detecting...")
    try:
        fid = dfe.represent(img_path = "./{}".format(img_name),model_name = models[7],detector_backend = backends[2])
    except:
        output["Result"]="Error..."
        print("Error: ")
    if(len(fid)==0):
        output["Result"]="No Face Detected"
        print("No Face Detected")
        return
    faceId = []
    faceId.append(img_name)
    faceId.append(fid)
    if(len(allFaces) !=0):
    # print("hg")
        try:
            checkFace = dfe.find(img_representation = fid,representations = allFaces,model_name = models[7],detector_backend = backends[2])
        except:
            output["Result"]="No Face Detected"
            print("No Face Detected")
        if(checkFace.empty):
            timeInfo[img_name] = time.time()
            allFaces.append(faceId)
            output["Result"]="Checkin Successful"
        else:
            output["Result"]="Face Already Exists"
            print("Face Already Exists")
    else:
        output["Result"]="Checkin Successful"
        timeInfo[img_name] = time.time()
        allFaces.append(faceId)
    # output.append("Number of People Checked In :- {}".format(len(allFaces)))
    output["Number"]= "Number of People Checked In :- {}".format(len(allFaces))
    print("Number of People Checked In :- {}".format(len(allFaces)))


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame,cin
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            if(cin==True or cout==True):
                global val, models,backends
                now = int(time.time())
                if(now%5 ==0 and now !=val):
                    val = now
                    img_name = ''.join(random.choices(string.ascii_uppercase +string.digits, k=50)) + ".png"
                    cv2.imwrite(img_name, frame)
                    if(cin==True and cout == True):
                        cin = False
                    if(cin):
                        checkin(img_name)
                    elif(cout):
                        checkout(img_name)
           
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/getData', methods=['POST'])
def get_data():
    global output
    out = output
    # if(cin == False):
    #     return json.dumps({})
    # output = {"Status":"",
    #       "Result":"",
    #       "Number":""}
    return json.dumps(out)

@app.route('/',methods=['POST','GET'])
def tasks():
    global switch,camera, output
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif request.form.get("checkin") == "CheckIn":
            global cin
            cin = not cin
        elif request.form.get("checkout") == "CheckOut":
            global cout
            if(cin==True):
                cin = not cin
                print("hello")
            cout = not cout
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                record(Frame)
                # thread = Thread(target = record, args=[out,])
                # thread.start()
            # elif(rec==False):
            #     out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html', variable=output) 
    return render_template('index.html', variable=output)


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     