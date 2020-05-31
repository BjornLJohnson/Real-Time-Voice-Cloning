from contentEncoder import model as ContentEncoder
import torch

ContentEncoder.load_model("./contentEncoder/saved_models/deepspeech5.pt", "./contentEncoder/saved_models/deepspeech6.pt", 8)
ContentEncoder.train()