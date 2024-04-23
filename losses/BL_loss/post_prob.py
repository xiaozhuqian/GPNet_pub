import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super(Post_Prob, self).__init__()
        #assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        hc_size, wc_size = c_size
        self.xcood = torch.arange(0, wc_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2 #torch.Size([160])
        self.xcood.unsqueeze_(0) #torch.Size([1, 160])
        self.ycood = torch.arange(0, hc_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2 
        self.ycood.unsqueeze_(0) #torch.Size([1, 288])
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points] #[24],batch size=1，图像上有24条鱼
        all_points = torch.cat(points, dim=0) #整个batch的点按行concate:torch.Size([24,2])

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1) #torch.Size([24, 1])
            y = all_points[:, 1].unsqueeze_(1) #torch.Size([24, 1])
            x_dis = -2 * torch.matmul(x, self.xcood) + x * x + self.xcood * self.xcood #torch.Size([24,160])
            y_dis = -2 * torch.matmul(y, self.ycood) + y * y + self.ycood * self.ycood #torch.Size([24,288])
            y_dis.unsqueeze_(2) #torch.Size([24,288,1])
            x_dis.unsqueeze_(1) #torch.Size([24,1,160])
            dis = y_dis + x_dis ##torch.Size([24,288,160])
            dis = dis.view((dis.size(0), -1)) #torch.Size([24,288*160])每个目标点距每个像素的距离

            dis_list = torch.split(dis, num_points_per_image) #tuple,len=bs,将每张图像分开，每个元素对应一个图像的所有目标点距离
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes): #循环一个batch内所有图像
                if len(dis) > 0: #图像上有目标
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)#torch.Size([1,288*160]),距每个像素最近的目标点的距离
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)#torch.Size([1,288*160])
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last,torch.Size([25, 46080])
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis) # 每个目标点在每个像素处出现的概率,当加上背景哑元，prob.shape=（N+1, h*w),N 为一张图像上目标点个数，h,w分别为该图像的高和宽
                else:
                    prob = None
                prob_list.append(prob) #返回一个batch的prob，每个元素为一张图像
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list


