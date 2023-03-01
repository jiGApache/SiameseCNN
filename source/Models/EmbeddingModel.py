from Models.SiameseModel import Siamese

class EmbeddingModule(Siamese):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.forward_once(x)