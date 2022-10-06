class Data:

    def getModels():
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
        return models

    def getBackends():
        backends = [
        'opencv', 
        'ssd', 
        'dlib', 
        'mtcnn', 
        'retinaface', 
        'mediapipe'
        ]
        return backends