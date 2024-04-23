import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from model import GPNet
import torch
import cv2
import time
import torch.nn as nn
import argparse
from torchvision import transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for test csrnet-attention', add_help=False)
    parser.add_argument('--data_root', default='/home/data/duanyunhong/datasets/arisdata',
                        help='path where the dataset is')
    parser.add_argument('--checkpoints_path', default='/home/data/duanyunhong/train_clear/best_model.pth',
                        help='path where to get trained model')
    
    parser.add_argument('--output_dir', default='/home/data/duanyunhong/train_clear/test',
                        help='path where to save, empty for no saving')

    parser.add_argument('--gpu_id', default='4', type=str, help='the gpu used for test')
    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

    root = args.data_root
    test_log_path = os.path.join(args.output_dir, 'test.txt')
    with open(test_log_path, "w") as f:
        f.write('Val Log %s\n' % time.strftime("%c"))

    #read pathes
    data_dir = os.path.join(root,'test','images')  
    img_paths = []
    for img_path in sorted(glob.glob(os.path.join(data_dir, '*.jpg'))):
        img_paths.append(img_path)

    #load trained weights
    model = GPNet()
    model = nn.DataParallel(model) #for multi-gpu trained weights
    model = model.cuda()
    checkpoint = torch.load(args.checkpoints_path)
    model.load_state_dict(checkpoint['state_dict']) 

    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])

    model.eval() 
    total_mae = 0
    total_mse = 0
    for i in range(len(img_paths)):
        _, image_name = os.path.split(img_paths[i])
        image_id = image_name.split('.')[0]
        img = Image.open(img_paths[i]).convert('RGB')
        img = transform(img)
        img = img.cuda()
        gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','groundtruths/density_map'),'r')
        groundtruth = np.asarray(gt_file['density']) #(576, 320)
        start_time = time.time()
        with torch.set_grad_enabled(False):
            outputs = model(img.unsqueeze(0))
            den_map = outputs[0]
            att_map = outputs[-1]

        time_one_image = time.time()-start_time
        predicted_count = den_map.detach().cpu().sum().numpy()
        groundtruth_count = np.sum(groundtruth)
        mae = abs(predicted_count-groundtruth_count)
        total_mae += mae
        mse = mae ** 2
        total_mse += mse
        print('img_path:',img_paths[i], 'mae=', mae, 'mse=', mse)

        den_map = den_map.squeeze(0).squeeze(0).detach().cpu().numpy() #(72, 40)
        att_map = att_map.squeeze(0).squeeze(0).detach().cpu().numpy() #(72, 40)
        postproc_den = post_proc(den_map)

        #save log
        with open(test_log_path, 'a') as f:
            f.write('image_id:{},groundtruth count:{:.2f},predicted count:{:.2f}, mae:{:.2f}, mse:{:.2f}, infer time:{:.3g}\n'\
                    .format(image_id,groundtruth_count,predicted_count,mae, mse, time_one_image))

        #save predicted and post processed density map
        pre_den_h5_path = os.path.join(args.output_dir, '{}_den.h5'.format(int(image_id)))
        with h5py.File(pre_den_h5_path, 'w') as hf:
            hf['den'] = den_map
        pre_den_jpg_path = os.path.join(args.output_dir, '{}_den.jpg'.format(int(image_id)))
        save_map(den_map, pre_den_jpg_path)
        proc_den_jpg_path = os.path.join(args.output_dir, '{}_den_postproc.jpg'.format(int(image_id)))
        save_map(postproc_den, proc_den_jpg_path)

        #save predicted attention map
        att_path = os.path.join(args.output_dir, '{}_att.jpg'.format(int(image_id)))
        save_map(att_map, att_path)
        
        #save visualizations
        save_path = os.path.join(args.output_dir, '{}_mae{:.2f}.jpg'.format(int(image_id),mae))
        save_result(int(image_id), save_path, Image.open(img_paths[i]),  groundtruth, den_map, postproc_den, att_map)

        #save gray predicted and gt density map
        # gray_pre_den =  np.maximum(output, 0)/np.max(output) * 255
        # gray_gt_den = np.maximum(groundtruth, 0)/np.max(groundtruth) * 255
        # gray_pre_den_path = os.path.join(args.output_dir, 'gray_pre_den','{}.jpg'.format(int(image_id)))
        # gray_gt_den_path = os.path.join(args.output_dir, 'gray_gt_den','{}.jpg'.format(int(image_id)))
        # cv2.imwrite(gray_pre_den_path, gray_pre_den)
        # cv2.imwrite(gray_gt_den_path, gray_gt_den)

    with open(test_log_path, 'a') as f:
        f.write('average mae: {:.2f}, average mse: {:.2f}\n'.format(total_mae/len(img_paths), (total_mse/len(img_paths)) ** 0.5))
    print('average mae:', total_mae/len(img_paths), 'average mse:', (total_mse/len(img_paths)) ** 0.5)

def post_proc(pre_den):
    # pre_den: original output from network
    gray_pre_den =  np.uint8(np.maximum(pre_den, 0)/np.max(pre_den) * 255)
    gau_pre_den = cv2.GaussianBlur(gray_pre_den,(7,7),2)

    return gau_pre_den

def save_map(map, path):
    plt.figure()
    plt.imshow(map,cmap=CM.jet)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight',pad_inches=0.0)

def save_result(img_id, path, img, gt_dmap, et_dmap, proc_dmap, att_map):
    '''draw input, groundtruth density map, predicted density map, postprocessed density map, predicted attention map on one image.
    
    img_id:input index, int
    path:save path
    img:input, PIL image
    gt_damp:groundtruth density map, ndarray(576,320)
    et_dmap:predicted density map, ndarray(576,320)
    proc_dmap: post processed density map, ndarray(576,320)
    att_map: predicted attention map
    '''
    plt.figure()
    fig, axes = plt.subplots(1, 5, figsize=(14, 6), sharex=True, sharey=True)
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

    ax[3].imshow(proc_dmap, cmap=CM.jet)
    ax[3].axis('off')
    ax[3].set_title('proc_dmap')

    ax[4].imshow(att_map, cmap=CM.jet)
    ax[4].axis('off')
    ax[4].set_title('att_map')

    fig.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('csrnet-attention test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)