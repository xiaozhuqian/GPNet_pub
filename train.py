import os
import visdom
import random
import numpy as np
import argparse
import time
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

from utils import Save_Handle

from datasets.crowd_sh import Crowd  #dataset
from model import GPNet

from losses.BL_loss.bay_loss import Bay_Loss
from losses.BL_loss.post_prob import Post_Prob
from losses import pytorch_ssim


parser = argparse.ArgumentParser(description='PyTorch GPNet')

#data
parser.add_argument('--data_dir', metavar='DATA_DIR', default='/home/data/duanyunhong/datasets/arisdata',
                    help='directory of data')

#output
parser.add_argument('--save_dir', default='/home/data/duanyunhong/train_clear', help='directory to saved model')
parser.add_argument('--max_model_num', type=int, default=1,
                        help='max models num to save ')
parser.add_argument('--vis_env',metavar='VIS_ENV', type=str, default='try',
                    help='Visdom environment name.') 

#train params
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')
parser.add_argument('--gpu',metavar='GPU', type=str, default='4,5',
                    help='GPU id to use.')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--original_lr', default=1e-5, type=float)
parser.add_argument('--steps', default=[-1, 1, 100, 150], nargs='+', type=int,
                    help='epoch to decay')
parser.add_argument('--scales', default=[1,1,1,1], nargs='+', type=int,
                    help='decay ratio')
parser.add_argument('--decay', default=1e-5, type=float)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--use_determin', default=1, type=int)

#loss
parser.add_argument('--seg_coe',metavar='SEG_COE', type=float, default='0.5',
                    help='segment loss coefficient.') 
parser.add_argument('--ssim_coe',metavar='SSIM_COE', type=float, default='100.0',
                    help='segment loss coefficient.') 
parser.add_argument('--sigma', type=float, default=4.0,
                    help='sigma for likelihood') #loss的超参
parser.add_argument('--background_ratio', type=float, default=10.0,
                    help='background ratio') #loss的超参


def main():
    
    global args,best_est_mae, train_log_path, val_log_path
    
    best_est_mae = 1e6
    best_epoch = 0
    
    args = parser.parse_args()
    args.workers = 1
    args.print_freq = 1

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 

    # log
    train_log_path = os.path.join(args.save_dir, 'train.txt')
    val_log_path = os.path.join(args.save_dir, 'val.txt')
    with open(train_log_path,'w') as f:
        f.write('Train Log %s\n' % time.strftime("%c"))
    with open(train_log_path, 'a') as f:
        f.write("{} \n".format(args))
    with open(val_log_path, "w") as f:
        f.write('Val Log %s\n' % time.strftime("%c"))

    # ensure reproducibility
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.use_determin:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True

    vis = visdom.Visdom(env = args.vis_env)

    # model
    model = GPNet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('number of params:', n_parameters)
    
    # loss
    ssim = pytorch_ssim.SSIM(window_size=11).cuda()
    bceloss = nn.BCELoss(reduction='sum').cuda()
    device = torch.device("cuda")
    post_prob = Post_Prob(args.sigma,
                          (288,160),
                          1,
                          args.background_ratio,
                          True,
                          device) #post probability
    criterion = Bay_Loss(True, device)
        
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # load pretrained weights
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_est_mae = checkpoint['best_est_mae']
            best_epoch = checkpoint['best_epoch'] 
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {} best mae {})"
                  .format(args.pre, checkpoint['epoch'], best_est_mae))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # number of saved checkpoints
    save_list = Save_Handle(max_num=args.max_model_num)

    start_time = time.time() 
    epochs = []
    train_losses = []
    train_den_losses = []
    train_seg_losses = []
    train_maes = []
    train_mses = []
    val_maes = []
    val_mses = []
    for epoch in range(args.start_epoch, args.epochs):
        #train
        adjust_learning_rate(optimizer, epoch)
        epoch_start_time = time.time()
        train_loss, train_den_loss, train_seg_loss, train_mae, train_mse = \
            train(model, ssim, post_prob, criterion, bceloss, optimizer)
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        #visualization train
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_den_losses.append(train_den_loss)
        train_seg_losses.append(train_seg_loss)
        train_maes.append(train_mae)
        train_mses.append(train_mse)
        vis.line(win=1, X=epochs, Y=train_losses, opts=dict(title='train_loss'))
        vis.line(win=2, X=epochs, Y=train_den_losses, opts=dict(title='train_den_losses'))
        vis.line(win=3, X=epochs, Y=train_seg_losses, opts=dict(title='train_seg_losses'))
        vis.line(win=4, X=epochs, Y=train_maes, opts=dict(title='train_mae'))
        vis.line(win=5, X=epochs, Y=train_mses, opts=dict(title='train_mse'))

        #val
        val_mae, val_mse = validate(model)
        epoch_total_time = time.time() - epoch_start_time
        
        #visualization val
        val_maes.append(val_mae)
        val_mses.append(val_mse)
        vis.line(win=6, X=epochs, Y=val_maes, opts=dict(title='val_mae'))
        vis.line(win=7, X=epochs, Y=val_mses, opts=dict(title='val_mse'))
        
        # best mae and epoch
        is_best = val_mae < best_est_mae
        best_est_mae = min(val_mae, best_est_mae)
        if is_best:
            best_epoch = epoch
        
        # save latest model
        save_path = os.path.join(args.save_dir, '{}_ckpt.tar'.format(epoch))
        torch.save({
            'epoch': epoch,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_est_mae': best_est_mae,
            'best_epoch': best_epoch,
            'optimizer' : optimizer.state_dict(),
        }, save_path)
        save_list.append(save_path)
        # save best model
        if is_best:
            torch.save({
            'epoch': epoch,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_est_mae': best_est_mae,
            'best_epoch': best_epoch,
            'optimizer' : optimizer.state_dict(),
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        #print and save log
        print('Epoch:{}, lr: {}, train_loss: {:.3f}, train_mae: {:.2f}, train_mse: {:.2f}, val_mae: {:.2f}, val_mse: {:.2f}, best mae: {:.3f}, best epoch: {}, epoch total time:{:.1f}'
            .format(epoch, lr, train_loss, train_mae, train_mse, val_mae, val_mse, best_est_mae, best_epoch, epoch_total_time))
        with open(train_log_path,'a') as f:
            f.write(f'epoch:{epoch}, lr:{optimizer.state_dict()["param_groups"][0]["lr"]},epoch_time:{epoch_total_time:.3f} s, train_total_loss:{train_loss:.2f}, train_den_loss {train_den_loss:.2f}, train_seg_loss {train_seg_loss:.2f}, train_MAE:{train_mae:.2f}, train_MSE:{train_mse:.2f}, val_MAE:{val_mae:.2f}, val_MSE:{val_mse:.2f}, best_epoch:{best_epoch}, best_val_MAE: {best_est_mae:.2f}\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('total time:{}'.format(total_time_str))
    with open(train_log_path,'a') as f:
        f.write('Total train_val time: {}\n'.format(total_time_str))

def train(model, ssim, post_prob, criterion, bceloss, optimizer):
    
    epoch_loss = AverageMeter() 
    epoch_den_loss = AverageMeter()
    epoch_seg_loss = AverageMeter()
    epoch_mse = AverageMeter()
    epoch_mae = AverageMeter()
    
    train_dataset = Crowd(os.path.join(args.data_dir, 'train'),
                                  crop_size=(288,160),
                                  downsample_ratio=1,
                                  is_gray=False, method='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    collate_fn=(train_collate),
                                    batch_size=(args.batch_size),
                                    shuffle=(True),
                                    num_workers=args.workers,
                                    pin_memory=(True))

    model.train()   
    for i,(img, points, targets, st_size, gt_den, gt_seg) in enumerate(train_loader):
        #send to gpu
        img = img.cuda()
        img = Variable(img)
        points = [p.cuda() for p in points]
        targets = [t.cuda() for t in targets]
        st_size = st_size.cuda()
        gt_den = gt_den.cuda()
        gt_den = Variable(gt_den)
        gt_seg = gt_seg.cuda()
        gt_seg = Variable(gt_seg)

        output = model(img)
        den_map = output[0] 
        att_map = output[-1]

        den_loss = 0 
        for den in output[:4]:
            prob_list = post_prob(points, st_size)
            bl_loss = criterion(prob_list, targets, den)
            ssim_loss = (1 - ssim(den, gt_den)) * args.ssim_coe
            den_loss += bl_loss + ssim_loss

        att_loss = bceloss(att_map, gt_seg) / img.size(0) * args.seg_coe
        loss = den_loss+att_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        #compute loss, mae, mse
        N = img.size(0) #batch
        pre_count = torch.sum(den_map.view(N, -1), dim=1).detach().cpu().numpy()
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        res = pre_count - gd_count
        epoch_loss.update(loss.item(), N) 
        epoch_seg_loss.update(att_loss.item(),N)
        epoch_den_loss.update(den_loss.item(),N)
        epoch_mse.update(np.mean(res * res), N)
        epoch_mae.update(np.mean(abs(res)), N) 
    
    return epoch_loss.avg, epoch_den_loss.avg, epoch_seg_loss.avg, epoch_mae.avg, np.sqrt(epoch_mse.avg)


def validate(model):
    val_dataset = Crowd(os.path.join(args.data_dir, 'val'),
                                  crop_size=(576,320),
                                  downsample_ratio=1,
                                  is_gray=False, method='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                    collate_fn=default_collate,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=False)   
    
    model.eval()
    mae = 0
    mse = 0
    for i,(img, count, name, gt_den, gt_seg, points) in enumerate(val_loader):
        img = img.cuda()
        img = Variable(img)
        outputs = model(img)
        val_den_map = outputs[0]
        
        val_res = val_den_map.data.sum()-count.cuda()
        mae += abs(val_res)
        mse += val_res*val_res
        
    mae = mae.cpu().numpy()/len(val_loader) 
    mse = np.sqrt(mse.cpu().numpy()/len(val_loader))    
  
    return mae.item(), mse.item()    
        
def adjust_learning_rate(optimizer, epoch):
    """ args.steps = [-1, epoch_1, epoch_2, epoch_3, ...]
        args.scales = [1, ratio_1, ratio_2, ratio_3, ...]
        when epoch_i <= epoch < epoch_j: 
            lr = original_lr * 1 * ratio_1 * ... * ratio_i

    """
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

def train_collate(batch):
    '''aggregate variables'''
    transposed_batch = list(zip(*batch)) 
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    gt_den = torch.stack(transposed_batch[4], 0)
    gt_seg = torch.stack(transposed_batch[5], 0)
    return images, points, targets, st_sizes, gt_den, gt_seg

if __name__ == '__main__':
    torch.set_num_threads(1)
    main()        