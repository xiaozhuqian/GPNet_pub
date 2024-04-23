import h5py
import torch
import shutil
import os
import numpy as np
from torchvision import datasets, transforms

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')   

class Save_Handle(object):
    """handle the number of """
    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)

def save_result(img, img_id, gt_dmap, et_dmap,path):
    from matplotlib import pyplot as plt
    from matplotlib import cm as CM
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    ax = axes.ravel()
 
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('image %d' % (img_id))
    
    ax[1].imshow(gt_dmap, cmap=CM.jet)
    ax[1].axis('off')
    ax[1].set_title('ground truth: %.2f' % (np.sum(gt_dmap)))
 
    ax[2].imshow(et_dmap, cmap=CM.jet)
    ax[2].axis('off')
    ax[2].set_title('estimated: %.2f' % (np.sum(et_dmap)))

    fig.tight_layout()
 
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.figure() 

def denorm(channel_mean, channel_std, img):
    '''
    tensor denorm
    input:
    img: tensor, eg:torch.Size([3, 288, 160])
    channel_mean: list, eg: [0.485, 0.456, 0.406]
    channel_std: list, eg: [0.229, 0.224, 0.225]
    output:
    img_r: denormed img; tensor, eg:torch.Size([3, 288, 160])
    '''
    MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
    STD = [1/std for std in channel_std]
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    img_r = denormalizer(img)
    return img_r
