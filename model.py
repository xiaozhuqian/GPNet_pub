import torch.nn as nn
import torch
from torchvision import models
from torchsummary import summary
from torchviz import make_dot
from att_danet.danet import DANetHead


class GPNet(nn.Module):
    'the size of input is better to match the FEM convolution kernel to get global receptive field.'
    def __init__(self, load_weights=False): 
        super(GPNet, self).__init__()
        self.seen = 0
        self.frontend = Backbone(bn=True)
        self.backend_att = DANetHead(in_channels = 512, out_channels = 1, norm_layer=nn.BatchNorm2d)
        self.att_last = nn.Sigmoid()
        self.backend_den = DME()
        self.den_last = BaseConv(in_channels=32, out_channels=1, kernel=1, stride=1, p=0, d_rate=1, activation=nn.ReLU(), use_bn=False)
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')

        if not load_weights:
            mod = models.vgg16_bn(pretrained = True)
            self._initialize_weights() 
            for i in range(70): #use pretrained vgg weights
                if 'num_batches_tracked' in list(self.frontend.state_dict().items())[i][0]: 
                    continue
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
            
    def forward(self,x):
        x = self.frontend(x)
        
        att_out_inter = self.backend_att(x[3])
        att_out_inter = list(att_out_inter)
        neck_out = self.att_last(att_out_inter[0]) * x[3]
        att_out = self.att_last(self.upsample(att_out_inter[0])) #pam+cam
        # att_out_inter[1] = self.upsample(att_out_inter[1]) #pam
        # att_out_inter[2] = self.upsample(att_out_inter[2]) #cam

        x[3] = neck_out
        den_out, den_0, den_1, den_2  = self.backend_den(x)
        den_out = self.den_last(den_out)

        return den_out, den_0, den_1, den_2, att_out

    def _initialize_weights(self):
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01) 
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) 
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
class Backbone(nn.Module):
    def __init__(self, bn=True, method='train'):
        super(Backbone, self).__init__()
        vgg_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        vgg = make_layers(vgg_feat,batch_norm=bn) 
        features = list(vgg.children())
        if bn:
            self.conv0 = nn.Sequential(*features[:6])
            self.maxp1 = features[6]
            self.conv1 = nn.Sequential(*features[7:13])  # 2x down sample 
            self.maxp2 = features[13]
            self.conv2 = nn.Sequential(*features[14:23]) # 4x down sample
            self.maxp3 = features[23] 
            self.conv3 = nn.Sequential(*features[24:33]) # 8x down sample 
            #self.body4 = nn.Sequential(*features[33:43]) # 16x down sample 
        else:
            self.conv0 = nn.Sequential(*features[:4])
            self.maxp1 = features[4]
            self.conv1 = nn.Sequential(*features[5:9])
            self.maxp2 = features[9]
            self.conv2 = nn.Sequential(*features[10:16])
            self.maxp3 = features[16]
            self.conv3 = nn.Sequential(*features[17:23])
            #self.body4 = nn.Sequential(*features[23:30])

        self.genet1 = FEM(channel=128, height=144, width=80)
        self.genet2 = FEM(channel=256, height=72, width=40)
        self.genet3 = FEM(channel=512, height=36, width=20)
        self.shotcut1 = BaseConv(in_channels=64, out_channels=128, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True)
        self.shotcut2 = BaseConv(in_channels=128, out_channels=256, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True)
        self.shotcut3 = BaseConv(in_channels=256, out_channels=512, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        out = []

        x_out0 = self.conv0(x)  #1x, 64
        
        x_conv1_in = self.maxp1(x_out0)
        x_conv1_out = self.conv1(x_conv1_in) #1/2x, 128
        x_out1 = x_conv1_out #1/2x, 128

        x_conv2_in = self.maxp2(x_out1)
        x_conv2_out = self.conv2(x_conv2_in)
        x_genet2_out = self.genet2(x_conv2_out)
        x_out2 = x_genet2_out + self.shotcut2(x_conv2_in) #1/4x, 128

        x_conv3_in = self.maxp3(x_out2)
        x_conv3_out = self.conv3(x_conv3_in)
        x_genet3_out = self.genet3(x_conv3_out)
        x_out3 = x_genet3_out + self.shotcut3(x_conv3_in) #1/8x, 128
        out = [x_out0, x_out1, x_out2, x_out3]

        return out

class DME(nn.Module):
    def __init__(self):
        super(DME, self).__init__()
        self.conv1 = nn.Sequential(BaseConv(in_channels=512, out_channels=256, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True),
                            BaseConv(in_channels=256, out_channels=256, kernel=3, stride=1, p = 2, d_rate = 2, activation=nn.ReLU(), use_bn=True),
                            nn.Upsample(scale_factor=2, mode='nearest'))

        self.conv2 = nn.Sequential(BaseConv(in_channels=512, out_channels=128, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True),
                            BaseConv(in_channels=128, out_channels=128, kernel=3, stride=1, p = 2, d_rate = 2, activation=nn.ReLU(), use_bn=True),
                            nn.Upsample(scale_factor=2, mode='nearest')) 

        self.conv3 = nn.Sequential(BaseConv(in_channels=256 , out_channels=64, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True),
                            BaseConv(in_channels=64, out_channels=64, kernel=3, stride=1, p = 2, d_rate = 2, activation=nn.ReLU(), use_bn=True),
                            nn.Upsample(scale_factor=2, mode='nearest')) 
        self.conv4 = nn.Sequential(BaseConv(in_channels=128, out_channels=32, kernel=1, stride=1, p = 0, d_rate = 1, activation=nn.ReLU(), use_bn=True),
                            BaseConv(in_channels=32, out_channels=32, kernel=3, stride=1, p = 2, d_rate = 2, activation=nn.ReLU(), use_bn=True))

        self.conv3_out = BaseConv(in_channels=64 , out_channels=1, kernel=3, stride=1, p = 1, d_rate = 1, activation=nn.ReLU(), use_bn=False)
        self.conv2_out = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                            BaseConv(in_channels=128 , out_channels=1, kernel=3, stride=1, p = 1, d_rate = 1, activation=nn.ReLU(), use_bn=False)
                            ) 
        self.conv1_out = nn.Sequential(nn.Upsample(scale_factor=4, mode='nearest'),
                            BaseConv(in_channels=256 , out_channels=1, kernel=3, stride=1, p = 1, d_rate = 1, activation=nn.ReLU(), use_bn=False)
                            )            
    
    def forward(self, x):
        c0=x[0]
        c1 = x[1]
        c2=x[2]
        c3=x[3]

        p2 = self.conv1(c3)  #1/4, 256
        p1 = self.conv2(torch.cat([p2, c2], 1)) #1/2, 128
        p0 = self.conv3(torch.cat([p1, c1], 1)) # 1, 64
        out = self.conv4(torch.cat([p0, c0], 1)) # 1, 1

        p0_out = self.conv3_out(p0)
        p1_out = self.conv2_out(p1)
        p2_out = self.conv1_out(p2)

        return out, p0_out, p1_out, p2_out

class FEM(nn.Module):
    def __init__(self, channel, height, width): 
        super(FEM, self).__init__()
        self.global_dwconv = nn.Sequential(nn.Conv2d(channel,
                                                    channel, 
                                                    kernel_size=(height,width), 
                                                    stride=(1,1), 
                                                    groups=channel, 
                                                    bias=False),
                                            #nn.BatchNorm2d(channel), 
                                            nn.AdaptiveAvgPool2d(1))   
        
        self.fc = nn.Sequential(nn.Linear(channel, channel // 16, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // 16, channel, bias=False),
                                nn.Sigmoid()) 

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_dwconv(x).view(b, c) #y.shape: (b, c)
        y = self.fc(y).view(b, c, 1, 1)#y.shape: (b, c, 1,1)
        return x * y.expand_as(x)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v== 'U':
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)    

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, p = 0, d_rate = 1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=p, dilation=d_rate)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input

#test
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = GPNet().cuda()
    summary(model, (3, int(576/2), int(320/2)))

    #visualize model
    input = torch.zeros(2, 3, 288, 160).cuda()
    output = model(input)
    print(output[0].shape, output[1].shape, output[2].shape,output[3].shape,output[-1].shape)
    g=make_dot(output)
    g.render('csrnet', view=False) #save as pdf
