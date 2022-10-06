from tabnanny import check
import cv2
import time
import DeepFace as dfe
from data import Data

def capture():
    models = Data.getModels()
    backends = Data.getBackends()
    checkIn = True
    startTime = time.time()
    allFaces = []
    timeInfo = {}
    FaceId = None
    img_counter = 0
    val = 0
    camera = cv2.VideoCapture(0) 
    cv2.namedWindow("test")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if k%256 == 32:
            checkIn = not checkIn
        nowTime = int(time.time())
        if  nowTime%5==0 and nowTime!=val:
            val = int(time.time())
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            if checkIn==True:
                print("Face Captured, Detecting...")
                try:
                    fid = dfe.represent(img_path = "./{}".format(img_name),model_name = models[7],detector_backend = backends[2])
                except:
                    print("Error: ")
                if(len(fid)==0):
                    print("No Face Detected")
                    continue
                faceId = []
                faceId.append(img_name)
                faceId.append(fid)
                if(len(allFaces) !=0):
                    # print("hg")
                    try:
                        checkFace = dfe.find(img_representation = fid,representations = allFaces,model_name = models[7],detector_backend = backends[2])
                    except:
                        print("No Face Detected")
                    if(checkFace.empty):
                        timeInfo[img_name] = time.time()
                        allFaces.append(faceId)
                    else:
                        print("Face Already Exists")
                else:
                    timeInfo[img_name] = time.time()
                    allFaces.append(faceId)
                print("Number of People Checked In :- {}".format(len(allFaces)))
                # checkIn(img_name) #--------------------------------------------------------------------------------------------------------------------------->
            else:
                print("Number of People Checked In :- {}".format(len(allFaces)))
                entry_time = 0
                exit_time = 0
                try:
                    fid = dfe.represent(img_path = "./{}".format(img_name),model_name = models[7],detector_backend = backends[2])
                except:
                    print("No Face Detected")
                    continue
                try:
                    checkFace = dfe.find(img_representation = fid,representations = allFaces,model_name = models[7],detector_backend = backends[2])
                except:
                    print("Error: Try again")
                    continue
                if(checkFace.empty):
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
                    print("CheckOut Complete")
                    print("Entry Time:- {}".format(entry_time))
                    print("Exit_time :- {}".format(exit_time))
                    print("Duration :- {}".format(exit_time-entry_time))
            img_counter += 1
    camera.release()
    cv2.destroyAllWindows()
capture()


