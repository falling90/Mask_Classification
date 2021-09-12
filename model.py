import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

# custom model
class Conv3x3BNReLU(nn.Module):
    '''
    ## Conv3x3BnRElu

     -subclass from nn.module
     -implement forward
     -kernel size 3,3 , bn-relu block
    '''
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        '''
                    -conv ->conv layer
                    -bn
        '''
        super(Conv3x3BNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        '''
        #### input : x
                input from previous layer
    
        #### output :

                output wiil be input to next layer

        '''
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=False)
    
class Conv1x1BNReLU(nn.Module):
    """
        ## Conv1x1BNReLU

        -subclass from nn.module
        -implement forward
        -kernel size 1,1 ,bn-relu block
    """
    def __init__(self, in_channels, out_channels):
        '''
            -conv -> conv layer
            -bn -> b atcnh normalization
        '''
        super(Conv1x1BNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
            
            ### forward


                #### input : x
                    input from previous layer

                #### output :

                    output wiil be input to next layer
        '''
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=False)

class MyModel(nn.Module):
    """
        ## MyModel

        -subclass from torch.utils.data.Dataset
        -implement len,getitem

    """
    def __init__(self, num_classes: int = 1000):
        '''
            -Conv1_k ,Conv1_k (k is integer)
                : conv 1*1 bnrelu
            -Conv k ( k is integer)
                : conv 3*3 bnrelu
            - Block k : (k is integer)
                : conv 1*1 bn-relu , conv 3*3 bn-relu
             - avg-pool : pooling layer

             -classifier : 
                 : output layer
         
        '''
        super(MyModel, self).__init__()
        
        self.Conv1_1 = Conv3x3BNReLU(in_channels=3, out_channels=32, stride=1, padding=1)
        self.Conv1_2 = Conv3x3BNReLU(in_channels=32, out_channels=64, stride=2)
        self.Block1 = nn.Sequential(
            Conv1x1BNReLU(64, 32),
            Conv3x3BNReLU(32, 64)
        )
        
        self.Conv2 = Conv3x3BNReLU(in_channels=64, out_channels=128, stride=2)
        self.Block2 = nn.Sequential(
            Conv1x1BNReLU(128, 64),
            Conv3x3BNReLU(64, 128)
        )
        
        self.Conv3 = Conv3x3BNReLU(in_channels=128, out_channels=256, stride=2)
        self.Block3 = nn.Sequential(
            Conv1x1BNReLU(256, 128),
            Conv3x3BNReLU(128, 256)
        )
        
        self.Conv4 = Conv3x3BNReLU(in_channels=256, out_channels=512, stride=2)
        self.Block4 = nn.Sequential(
            Conv1x1BNReLU(512, 256),
            Conv3x3BNReLU(256, 512)
        )
        
        self.Conv5 = Conv3x3BNReLU(in_channels=512, out_channels=1024, stride=2)
        self.Block5 = nn.Sequential(
            Conv1x1BNReLU(1024, 512),
            Conv3x3BNReLU(512, 1024)
        )        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        '''
        
        
        ### forward


            #### input : x
                input image

            #### output :

                output ,  softmax multilabel classification   

        
        '''
        x = self.Conv1_1(x)
        x = self.Conv1_2(x)
        x_temp = x.clone()
        x = self.Block1(x)
        x += x_temp
        
        x = self.Conv2(x)
        for i in range(2):
            x_temp = x.clone()
            x = self.Block2(x)
            x += x_temp
        
        x = self.Conv3(x)
        for i in range(8):
            x_temp = x.clone()
            x = self.Block3(x)
            x += x_temp
        
        x = self.Conv4(x)
        for i in range(8):
            x_temp = x.clone()
            x = self.Block4(x)
            x += x_temp
        
        x = self.Conv5(x)
        for i in range(4):
            x_temp = x.clone()
            x = self.Block5(x)
            x += x_temp
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

def config_model(model_name):
    """
    ## Config Model

    ###     input : model_name
    ####       specific model name

    ####     pre_classified_models : model_list



    ###      output : model 
    ####            pretrained model from hub
    
    """
    pre_classified_models = ['vit_base_patch16_224','vit_large_patch16_224','vgg16','custom_model']
    if model_name in pre_classified_models:
        if model_name == 'vit_base_patch16_224':
            model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=18)
        elif model_name == 'vit_large_patch16_224':
            model = timm.create_model('vit_large_patch16_224',pretrained=True,num_classes=18)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'custom_model':
            model = MyModel(num_classes=18)
    else :
        if model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=True)
        elif model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True) 
        num_features = model.fc.out_features
        model.fc = nn.Linear(num_features, 18)
    return model