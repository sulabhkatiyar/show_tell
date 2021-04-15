import torch
from torch import nn
import torchvision
import pretrainedmodels
import json
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""HERE WE HAVE CHOICE OF ENCODER. 0 for VGG-16; more to be added soon."""

encoder_choice = 0

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=1):
        super(Encoder, self).__init__()        
        self.enc_image_size = encoded_image_size

        if encoder_choice==0:
            self.vgg16 = torchvision.models.vgg16(pretrained = True)
            self.features_nopool = nn.Sequential(*list(vgg16.features.children())[:-1])
            self.features_pool = list(vgg16.features.children())[-1]
            self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1]) 

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        
        global encoder_choice
        
        if encoder_choice==0: 
            x = self.features_nopool(images)
            x_pool = self.features_pool(x)
            x_feat = x_pool.view(x_pool.size(0), -1)
            y = self.classifier(x_feat)
            return y.unsqueeze(1).unsqueeze(1)
            
        out = self.adaptive_pool(out)  # (batch_size, 512, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if encoder_choice==0:
            for p in self.vgg16.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks after 20

            for c in list(self.vgg.children())[0][20:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

