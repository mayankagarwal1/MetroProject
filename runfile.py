import re
import DeepFace

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
e = DeepFace.represent(img_path = "./p1/b.jpg", 
      model_name = models[7],detector_backend = backends[2]
)


# # e1 = DeepFace.represent(img_path = "./p1/a.jpg", 
# #       model_name = models[7],detector_backend = backends[2]
# # )
# # print(len(e))
# # print(len(e1))
# # result = DeepFace.verify(img1_path = "./p1/d.jpg", img2_path = "./p1/f.jpg", model_name = models[0],detector_backend = backends[2])
# # result = DeepFace.verify(e, e1, model_name = models[7],detector_backend = backends[2])

# representations = DeepFace.representAll(db_path = "./p1",model_name = models[7],detector_backend = backends[2])
# df = DeepFace.find(img_path = "./p1/c.jpg",
#       db_path = "./p1", 
#       model_name = models[7],detector_backend = backends[2]
# )
# df = DeepFace.find(img_representation=e,
#       representations = representations, 
#       model_name = models[7],detector_backend = backends[2]
# )
# print(type(df))
# print(df)
# print(representations)
exit();