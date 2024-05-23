import torch.nn as nn
import torchvision.models as models


class CnnFeatureExtractor(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        if model_name == "vgg11":
            base_model = models.vgg11()
        if model_name == "vgg13":
            base_model = models.vgg13_bn()
        if model_name == "vgg11":
            base_model = models.vgg16_bn()
        elif model_name == "vgg19":
            base_model = models.vgg19_bn()
        self.layers = nn.Sequential(*list(base_model.children())[:-1])
        self.out_features = base_model.classifier[0].in_features

    def forward(self, x):
        h = self.layers(x)
        h = h.flatten(start_dim=1)
        return h

class DeconvDecoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_size, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1) # from [batch, features] to [batch, features, 1, 1]
        y_hat = self.layers(x)
        return y_hat

class HighDimModel(nn.Module):
    def __init__(self, model_name: nn.Module, latent_size: int, num_classes: int):
        super().__init__()
        self.cnn = CnnFeatureExtractor(model_name, num_classes) #extracts features from input image
        self.fc = nn.Sequential(nn.BatchNorm1d(self.cnn.out_features), # turn features into 1d latent representation
                                nn.LeakyReLU(),
                                nn.Linear(self.cnn.out_features, latent_size))
        self.decoder = DeconvDecoder(latent_size)
        self.model_name = f"{model_name}_highdim"
        self.num_classes = num_classes

    def forward(self, x):
        h = self.cnn(x)
        lr = self.fc(h)
        y_hat = self.decoder(lr)
        y_hat = y_hat.view(-1, 64, 64)
        return y_hat


class CategoricalModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.cnn = CnnFeatureExtractor(model_name, num_classes) # extracts features from input image
        self.fc = nn.Linear(self.cnn.out_features, num_classes)
        self.model_name = f"{model_name}_categorical"
        self.num_classes = num_classes

    def forward(self, data):
        x = self.cnn(data)
        x = x.view(x.size(0), -1)
        prediction = self.fc(x)
        return prediction


class BaselineModel(nn.Module):
    def __init__(self, model_name: str, num_classes=10):
        super().__init__()
        if model_name == "vgg11":
            base_model = models.vgg11_bn()
        if model_name == "vgg13":
            base_model = models.vgg13_bn()
        if model_name == "vgg16":
            base_model = models.vgg16_bn()
        elif model_name == "vgg19":
            base_model = models.vgg19_bn()
        self.base_model = base_model.features # need to edit about the base number of classes
        flatten_features = 512
        self.fc = nn.Linear(flatten_features, 10)

    def forward(self, data):
        y = self.base_model(data)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y