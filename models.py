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
            vgg16 = torchvision.models.vgg16(pretrained = True)
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
            out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        elif encoder_choice==1:
            out = self.incepres(images) # (batch_size, 1536, image_size/32, image_size/32)
        elif encoder_choice==2:
            out = self.nasnetlarge(images)  # (batch_size, 4032, image_size/32, image_size/32)
        elif encoder_choice==3:
            out = self.vgg(images)     # (batch_size, 512, image_size/32, image_size/32)
        elif encoder_choice==4:
            out = self.alexnet(images)     # (batch_size, 256, 6, 6)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 6, 6, 256)
            return out
        elif encoder_choice == 5:
            out = self.squeezenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 15, 15, 512)
            return out
        elif encoder_choice == 6:
            out = self.densenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1920) 
            return out
        elif encoder_choice == 7:
            out = self.googlenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1024) 
            return out
        elif encoder_choice == 8:
            out = self.shufflenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1024) 
            return out
        elif encoder_choice == 9:
            out = self.mobilenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1280) 
            return out
        elif encoder_choice == 10:
            out = self.resnext(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 11:
            out = self.wideresnet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 12:
            out = self.mnasnet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1280) 
            return out
        elif encoder_choice == 13:
            out = self.xception(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 14:
            out = self.inception(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1536) 
            return out
        elif encoder_choice == 15:
            out = self.nasnetamobile(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1056) 
            return out
        elif encoder_choice == 16:
            out = self.dpn131(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2688) 
            return out
        elif encoder_choice == 17:
            out = self.senet154(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 18:
            out = self.pnasnet5large(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 4320) 
            return out
        elif encoder_choice == 19:
            out = self.polynet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 6, 6, 2048) 
            return out
        elif encoder_choice == 20:
            out = self.resnet18(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 512) 
            return out
        elif encoder_choice == 21:
            out = self.resnet34(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 512) 
            return out
        elif encoder_choice == 22:
            out = self.resnet50(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 23:
            out = self.resnet101(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 24:
            out = self.vgg11(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512) 
            return out
        elif encoder_choice == 25:
            out = self.vgg13(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512) 
            return out
        elif encoder_choice == 26:
            out = self.vgg19(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512) 
            return out
        elif encoder_choice == 27:
            out = self.densenet121(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1024) 
            return out
        elif encoder_choice == 28:
            out = self.densenet169(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1664) 
            return out
        elif encoder_choice == 29:
            out = self.densenet161(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2208) 
            return out
        elif encoder_choice==30:
            out = self.vgg16_nobn(images)     # (batch_size, 512, image_size/32, image_size/32)
        elif encoder_choice==31:
            # print(images.shape)
            x = self.features_nopool(images)
            # print(x.shape)
            x_pool = self.features_pool(x)
            # print(x_pool.shape)
            x_feat = x_pool.view(x_pool.size(0), -1)
            # print(x_feat.shape)
            y = self.classifier(x_feat)
            # print(y.shape)
            return y.unsqueeze(1).unsqueeze(1)
            
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # if type(images) == list:
        #     out = self.activation(self.combine(out))
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if encoder_choice==0:
            for p in self.resnet.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif encoder_choice==1:
            for p in self.incepres.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.incepres.children())[10:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif encoder_choice==2:
            for p in self.naslarge_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.naslarge_model.children())[15:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif encoder_choice==3:
            for p in self.vgg.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks after 20
            for c in list(self.vgg.children())[0][20:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

