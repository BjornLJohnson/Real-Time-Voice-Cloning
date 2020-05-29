from contentEncoder import model as ContentEncoder
import torch

filePath = "./data/contentAudio/121-121726-0010.flac"

ContentEncoder.load_model("./contentEncoder/saved_models/deepspeech4.pt")
# ContentEncoder.train()
# print(ContentEncoder._model)
print(ContentEncoder.encode(filePath))
