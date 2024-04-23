from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import h5py
import cv2

def random_crop(img, den_map, seg_map, points, wc_size, hc_size):
    '''
    input:
        img: tensor(channel, h, w)
        seg_map: tensor(channel, h, w)
        den_map: tensor(channel, h, w)
        points: ndarray[x,y](count,2)
    return:
        result_img: tensor(channel, hc_size, wc_size)
        seg_map: tensor(channel, hc_size, wc_size)
        den_map: tensor(channel, hc_size, wc_size)
        record_points: ndarray[x,y](count,2)
    '''
    half_h = hc_size
    half_w = wc_size
    result_img = np.zeros([img.shape[0], half_h, half_w])
    result_den_map = np.zeros([img.shape[0], half_h, half_w])
    result_seg_map = np.zeros([img.shape[0], half_h, half_w])
 
    # crop num_patch for each image
    start_h = random.randint(0, img.size(1) - half_h)
    start_w = random.randint(0, img.size(2) - half_w)
    end_h = start_h + half_h
    end_w = start_w + half_w
    # copy the cropped rect
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_den_map = den_map[:, start_h:end_h, start_w:end_w]
    result_seg_map = seg_map[:, start_h:end_h, start_w:end_w]
    # copy the cropped points
    if len(points)>0:
        idx = (points[:, 0] >= start_w) & (points[:, 0] <= end_w) & (points[:, 1] >= start_h) & (points[:, 1] <= end_h)
        # shift the corrdinates
        record_points = points[idx]
        record_points[:, 0] -= start_w
        record_points[:, 1] -= start_h
    else: #when count=0:
        record_points = points

    return result_img, result_den_map, result_seg_map, record_points   

class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):
        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path,'images', '*.jpg')))
        if method=='train':
            self.im_list = self.im_list *4  
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.hc_size, self.wc_size = crop_size
        self.d_ratio = downsample_ratio
        
        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        '''
        train return:
            image:croped and flipped image(PIL); 
            keypoints:croped and flipped target coordinates (ndarray)
            target:tensor[1,1,...,1],len=count; 
            st_size:min(h, w) of original image
        val return: 
            img: totensor,normalize; 
            count; 
            name: image id.
        test return:
            same with val
        '''
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'txt').replace('images','groundtruths/points')
        den_path = img_path.replace('jpg', 'h5').replace('images','groundtruths/density_map')
        seg_path = img_path.replace('images','groundtruths/fish_mask').replace('.jpg','.h5')
        #load images
        img = Image.open(img_path).convert('RGB')
        #load points coordinates
        keypoints = []
        with open(gd_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                keypoints.append([x, y])
        #load groundtruth density map
        gt_den_file = h5py.File(den_path)
        gt_den = np.asarray(gt_den_file['density'])
        #load seg map
        gt_seg_file = h5py.File(seg_path)
        gt_seg = np.asarray(gt_seg_file['seg'], dtype=np.float32)
        gt_seg = gt_seg/255.

        if self.method == 'train':
            #random crop, random horizontal flip
            img, points, targets, st_size, gt_den, gt_seg = self.train_transform(img, np.array(keypoints), gt_den, gt_seg) #targets: count, st_size: min(h,w)
            return img, points, targets, st_size, gt_den, gt_seg #torch.Size([3, 160, 160]) torch.Size([4, 2]) torch.Size([4]) 320 torch.Size([1, 160, 160])
        elif self.method == 'val':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name, torch.from_numpy(gt_den.copy()), torch.from_numpy(gt_seg.copy()), keypoints

    def train_transform(self, img, keypoints, den_map, seg_map):
        """random crop image patch and find people in it
        """
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= min(self.hc_size, self.wc_size)
        
        # random crop
        img, den_map, seg_map, keypoints = random_crop(transforms.ToTensor()(img), torch.from_numpy((den_map[np.newaxis,:]).copy()), torch.from_numpy((seg_map[np.newaxis,:]).copy()), keypoints, self.wc_size, self.hc_size)
        target = np.ones(len(keypoints))
        img = transforms.ToPILImage()(img)
        #random flip
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                den_map = F.hflip(den_map)
                seg_map = F.hflip(seg_map)
                keypoints[:, 0] = self.wc_size - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                den_map = F.hflip(den_map)
                seg_map = F.hflip(seg_map)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size, den_map, seg_map

def draw_img_point(image, image_name, points, out_dir):
    '''
    将points画到image上并保存
    image: PIL image
    image_name: str
    points: ndarray[x,y](count,2)
    out_dir: str
    '''
    # draw the predictions
    size = 1
    img_to_draw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)    
    # save the visualized predicted images
    #img_index = image_name.split('.',1)[0]
    count = len(points)
    cv2.imwrite(os.path.join(out_dir, image_name), img_to_draw)   

# test the code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm as CM

    root = '/home/duanyunhong/datasets/arisdata_full'
    #test train
    train_dataset_path = os.path.join(root, 'train')
    train_dataset = Crowd(train_dataset_path, crop_size=(288, 160),
                 downsample_ratio=1, is_gray=False,
                 method='train'
                )
    for i, (img, points, targets, st_size, gt_den, gt_seg) in enumerate(train_dataset):
        if i>5:
            break
        print('i=', i, 'dataset', img.shape, points.shape, targets.shape, st_size, gt_den.shape, gt_seg.shape) #torch.Size([3, 288, 160]) torch.Size([7, 2]) torch.Size([7]) 320 torch.Size([1, 288, 160]) torch.Size([1, 288, 160])
        img[0] = img[0]*0.229+0.485
        img[1] = img[1]*0.224+0.456
        img[2] = img[2]*0.225+0.406
        to_PIL = transforms.ToPILImage()
        img = to_PIL(img)
        gt_den = gt_den.squeeze(0).numpy()
        gt_seg = np.transpose((gt_seg.numpy())*255., (1, 2, 0))
        name = os.path.basename(train_dataset.im_list[i]).split('.')[0]
        draw_img_point(img, '{}_count_train.jpg'.format(i), points, 'vis')
        plt.figure()
        plt.imshow(gt_den, cmap=CM.jet)
        plt.axis('off')
        plt.title('ground truth: %.2f' % (np.sum(gt_den)))
        path = os.path.join('vis','{}_den_train.jpg'.format(i))
        plt.savefig(path, bbox_inches='tight')
        plt.show()
        cv2.imwrite(os.path.join('vis','{}_seg_train.jpg'.format(i)), gt_seg)
    

    #test val
    val_dataset_path = os.path.join(root, 'val')
    
    val_dataset = Crowd(val_dataset_path, crop_size=(576, 320),
                 downsample_ratio=1, is_gray=False,
                 method='val'
                )
    for i, (img, count, name, gt_den, gt_seg, points) in enumerate(val_dataset):
        if i>2:
            break
        print('val','i=', i, 'img.shape:', img.shape, 'count:', count, 'name:', name, 'gt_seg.shape:', gt_seg.shape, 'points_len:', len(points)) #img.shape: torch.Size([3, 576, 320]) count: 0 name: 0 gt_seg.shape: torch.Size([576, 320]) points_len: 0
        img[0] = img[0]*0.229+0.485
        img[1] = img[1]*0.224+0.456
        img[2] = img[2]*0.225+0.406
        to_PIL = transforms.ToPILImage()
        img = to_PIL(img)
        gt_den = gt_den.squeeze(0).numpy()
        gt_seg = (gt_seg.numpy())*255.
        draw_img_point(img, '{}_count_val.jpg'.format(name), points, 'vis')
        plt.figure()
        plt.imshow(gt_den, cmap=CM.jet)
        plt.axis('off')
        plt.title('ground truth: %.2f' % (np.sum(gt_den)))
        path = os.path.join('vis','{}_den_val.jpg'.format(name))
        plt.savefig(path, bbox_inches='tight')
        plt.show()
        cv2.imwrite(os.path.join('vis','{}_seg_val.jpg'.format(name)), gt_seg)
