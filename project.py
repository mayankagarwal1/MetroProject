import cv2
import time
import DeepFace as dfe
from data import Data

class Capture:
    models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "DeepFace", 
        "DeepID", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        ]
    backends = [
        'opencv', 
        'ssd', 
        'dlib', 
        'mtcnn', 
        'retinaface', 
        'mediapipe'
        ]
    def __init__(self) -> None:
        self.checkIn = True
        self.startTime = time.time()
        self.allFaces = []
        self.timeInfo = {}
        self.FaceId = None
        self.img_counter = 0
        self.val = 0


    def checkInFace(self,img_name):
        print("Face Captured, Detecting...")
        try:
            fid = dfe.represent(img_path = "./{}".format(img_name),model_name = self.models[7],detector_backend = self.backends[2])
        except:
            print("Error: ")
            return
        if(len(fid)==0):
            print("No Face Detected")
            return
        faceId = []
        faceId.append(img_name)
        faceId.append(fid)
        if(len(self.allFaces) !=0):
            # print("hg")
            try:
                checkFace = dfe.find(img_representation = fid,representations = self.allFaces,model_name = self.models[7],detector_backend = self.backends[2])
            except:
                print("No Face Detected")
                return
            if(checkFace.empty):
                self.timeInfo[img_name] = time.time()
                self.allFaces.append(faceId)
            else:
                print("Face Already Exists")
        else:
            self.timeInfo[img_name] = time.time()
            self.allFaces.append(faceId)
        print("Number of People Checked In :- {}".format(len(self.allFaces)))

    def checkOut(self,img_name):
        print("Number of People Checked In :- {}".format(len(self.allFaces)))
        entry_time = 0
        exit_time = 0
        try:
            fid = dfe.represent(img_path = "./{}".format(img_name),model_name = self.models[7],detector_backend = self.backends[2])
        except:
            print("No Face Detected")
            return
            # continue
        try:
            checkFace = dfe.find(img_representation = fid,representations = self.allFaces,model_name = self.models[7],detector_backend = self.backends[2])
        except:
            print("Error: Try again")
            return
            # continue
        if(checkFace.empty):
            print("Face Not Found or Never Checked In")
        f = 0
        for i in checkFace['identity']:
            for j in self.allFaces:
                if(i==j[0]):
                    entry_time = self.timeInfo[i]
                    self.allFaces.remove(j)
                    f=1
        exit_time = time.time();
        if(f==1):
            print("CheckOut Complete")
            print("Entry Time:- {}".format(entry_time))
            print("Exit_time :- {}".format(exit_time))
            print("Duration :- {}".format(exit_time-entry_time))
            return True
        return False

    def captureFace(self):
        camera = cv2.VideoCapture(0) 
        # cv2.namedWindow("test")
        while True:
            ret, frame = camera.read()
            if not ret:
                print("failed to grab frame")
                break
            # cv2.imshow("test", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            if k%256 == 32:
                self.checkIn = not self.checkIn
            nowTime = int(time.time())

            if  nowTime%5==0 and nowTime!=self.val:

                self.val = int(time.time())
                img_name = "opencv_frame_{}.png".format(self.img_counter)
                cv2.imwrite(img_name, frame)

                if self.checkIn==True:
                    self.checkInFace(img_name)

                else:
                    self.checkOut(img_name=img_name)

                self.img_counter += 1
        camera.release()
        cv2.destroyAllWindows()