import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def pixel_unshuffle(fm, r):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    b, c, h, w = fm.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
    fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(b, out_channel, out_h, out_w)

    return fm_prime


class VarLoss(nn.Module):
    def __init__(self, depth_channel, feat_channel):
        super(VarLoss, self).__init__()

        self.att = nn.Sequential(
                nn.Conv2d(feat_channel, depth_channel, kernel_size=3, padding=1),
                nn.Sigmoid())

        self.post = nn.Conv2d(depth_channel, 2, kernel_size=3, padding=1)

        self.r = 10  # repeat sample

    def forward(self, x, d, gts):
        loss = 0.0
        for i in range(self.r):
            loss = loss + self.single(x, d, gts)

        return loss / self.r

    def single(self, feat, d, gts):

        ts_shape = d.shape[2:]
        gt = gts.clone()
        #gt = gts.unsqueeze(1)
        #print(gt.shape)
        os_shape = gt.shape[2:]
        n, c, h, w = d.shape

        reshaped_gt, indices = self.random_pooling(gt, ts_shape)

        bias_x = os_shape[1] // ts_shape[1] // 2
        bias_y = os_shape[0] // ts_shape[0] // 2 * os_shape[1]

        indices = indices + bias_x + bias_y
        ind_x = (indices % os_shape[1]).to(d.dtype) / os_shape[1]
        ind_y = (indices // os_shape[1]).to(d.dtype) / os_shape[0]

        ind_x = 2 * (ind_x - 0.5)
        ind_y = 2 * (ind_y - 0.5)
        grid = torch.cat([ind_x, ind_y], 1)
        grid = grid.permute(0, 2, 3, 1)

        feat = F.grid_sample(input=feat, grid=grid, mode='bilinear', align_corners=True)

        att = self.att(feat)

        ds = att * d
        #att = att.permute(0, 2, 3, 1)

        #ds = F.grid_sample(input=d, grid=grid+att, mode='bilinear', align_corners=True)

        ds = self.post(ds)

        loss = self.loss(ds, reshaped_gt)
        return loss
    
    def random_pooling(self, gt_depth, shape):
        #print(gt_depth.shape)
        n, c, h, w = gt_depth.shape
        rand = torch.rand(n, c, h, w, dtype=gt_depth.dtype, device=gt_depth.device)
        mask = gt_depth > 0.1
        rand = rand * mask

        _, indices = F.adaptive_max_pool2d(rand, shape, return_indices=True)

        reshaped_ind = indices.reshape(n, c, -1)
        reshaped_gt = gt_depth.reshape(n, c, h*w)
        reshaped_gt = torch.gather(input=reshaped_gt, dim=-1, index=reshaped_ind)
        reshaped_gt = reshaped_gt.reshape(n, c, indices.shape[2], indices.shape[3])

        reshaped_gt[reshaped_gt < 0.1] = 0
        return reshaped_gt, indices

    def grad(self, image):
        def gradient_y(img):
            gx = torch.log(img[:,:,1:-1,1:-1]+1e-6) - torch.log(img[:,:,2:,1:-1]+1e-6)

            mask = img > 0.1
            mask = torch.logical_and(mask[:,:,1:-1,1:-1], mask[:,:,2:,1:-1])
            return gx, mask

        def gradient_x(img):
            gy = torch.log(img[:,:,1:-1,1:-1]+1e-6) - torch.log(img[:,:,1:-1,2:]+1e-6)

            mask = img > 0.1
            mask = torch.logical_and(mask[:,:,1:-1,1:-1], mask[:,:,1:-1,2:])
            return gy, mask

        image = F.pad(image, (1,1,1,1), 'constant', 0.0)

        image_grad_x, mask_x = gradient_x(image)
        image_grad_y, mask_y = gradient_y(image)

        return image_grad_x, image_grad_y, mask_x, mask_y

    def loss(self, ds, reshaped_gt):

        gx, gy, mx, my = self.grad(reshaped_gt)
        grad_gt = torch.cat([gx, gy], 1)
        grad_mk = torch.cat([mx, my], 1)

        diff = F.smooth_l1_loss(ds, grad_gt, reduce=False, beta=0.01) * grad_mk

        loss_g =  diff.sum() / grad_mk.sum()

        return loss_g


class WindowVarLoss(nn.Module):
    def __init__(self, depth_channel, feat_channel, window_size=1):
        super(WindowVarLoss, self).__init__()

        self.att = nn.Sequential(
                nn.Conv2d(feat_channel, depth_channel, kernel_size=3, padding=1),
                nn.Sigmoid())
        
        self.window = window_size

        self.post = nn.Conv2d(depth_channel, 4 * self.window, kernel_size=3, padding=1)

        self.r = 5 * 4 * self.window  # repeat sample

    def forward(self, x, d, gts):
        loss = 0.0
        for i in range(self.r):
            loss = loss + self.single(x, d, gts)

        return loss / self.r

    def single(self, feat, d, gts):

        ts_shape = d.shape[2:]
        gt = gts.clone()
        os_shape = gt.shape[2:]
        n, c, h, w = d.shape

        reshaped_gt, indices = self.random_pooling(gt, ts_shape)

        bias_x = os_shape[1] // ts_shape[1] // 2
        bias_y = os_shape[0] // ts_shape[0] // 2 * os_shape[1]

        indices = indices + bias_x + bias_y
        ind_x = (indices % os_shape[1]).to(d.dtype) / os_shape[1]
        ind_y = (indices // os_shape[1]).to(d.dtype) / os_shape[0]

        ind_x = 2 * (ind_x - 0.5)
        ind_y = 2 * (ind_y - 0.5)
        grid = torch.cat([ind_x, ind_y], 1)
        grid = grid.permute(0, 2, 3, 1)

        feat = F.grid_sample(input=feat, grid=grid, mode='bilinear', align_corners=True)

        att = self.att(feat)

        ds = att * d
        #ds = F.grid_sample(input=d, grid=grid+att, mode='bilinear', align_corners=True)

        ds = self.post(ds)

        loss = self.loss(ds, reshaped_gt)
        return loss
    
    def random_pooling(self, gt_depth, shape):
        n, c, h, w = gt_depth.shape
        rand = torch.rand(n, c, h, w, dtype=gt_depth.dtype, device=gt_depth.device)
        mask = gt_depth > 0.1
        rand = rand * mask

        _, indices = F.adaptive_max_pool2d(rand, shape, return_indices=True)

        reshaped_ind = indices.reshape(n, c, -1)
        reshaped_gt = gt_depth.reshape(n, c, h*w)
        reshaped_gt = torch.gather(input=reshaped_gt, dim=-1, index=reshaped_ind)
        reshaped_gt = reshaped_gt.reshape(n, c, indices.shape[2], indices.shape[3])

        reshaped_gt[reshaped_gt < 0.1] = 0
        return reshaped_gt, indices

    def grad(self, image, window):
        grads = []
        masks = []
        mask = image > 0.1
        for d in range(1,window+1):
            for direction in range(0,4):
                if direction == 0:
                    ox, oy = 0, d
                elif direction == 1:
                    ox, oy = d, d
                elif direction == 2:
                    ox, oy = d, 0
                else:
                    ox, oy = d, -d
                
                last_x = -window+ox
                if last_x == 0:
                    last_x = 100000
                last_y = -window+oy
                if last_y == 0:
                    last_y = 100000
                g = torch.log(image[:,:,window:-window,window:-window]+1e-6) - torch.log(image[:,:,window+ox:last_x,window+oy:last_y]+1e-6)
                m = torch.logical_and(mask[:,:,window:-window,window:-window], mask[:,:,window+ox:last_x,window+oy:last_y])
                grads.append(g)
                masks.append(m)
        grads = torch.cat(grads, 1)
        masks = torch.cat(masks, 1)
        return grads, masks

    def loss(self, ds, reshaped_gt):
        image = F.pad(reshaped_gt, (self.window,self.window,self.window,self.window), 'constant', 0.0)
        grad_gt, grad_mk = self.grad(image, self.window)

        diff = F.smooth_l1_loss(ds, grad_gt, reduce=False, beta=0.01) * grad_mk

        loss_g =  diff.sum() / grad_mk.sum()

        return loss_g


class SILogLoss(nn.Module):
    def __init__(self, SI_loss_lambda, max_depth):
        super(SILogLoss, self).__init__()

        self.SI_loss_lambda = SI_loss_lambda
        self.max_depth = max_depth

    def forward(self, feat, gts):

        loss = 0

        for key in feat:

            depth_prediction = feat[key]
            shape = [depth_prediction.shape[2], depth_prediction.shape[3]]
            #scale_factor = self.shape_h // shape[0]
            scale_factor = int(np.sqrt(depth_prediction.shape[1]))

            reshaped_gt = pixel_unshuffle(gts, scale_factor)

            diff = torch.log(depth_prediction) - torch.log(reshaped_gt)

            num_pixels = (reshaped_gt > 0.1) * (reshaped_gt < self.max_depth)

            diff = torch.where(
            (reshaped_gt > 0.1) * (reshaped_gt < self.max_depth) * (torch.abs(diff) > 0.001),
            diff,
            torch.zeros_like(diff)
            )
            lamda = self.SI_loss_lambda

            diff = diff.reshape(diff.shape[0], -1)
            num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

            loss1 = (diff**2).sum(dim=-1) / num_pixels
            loss1 = loss1 - lamda * (diff.sum(dim=-1) / num_pixels) ** 2
            #loss1 = diff.abs().sum(dim=-1) / num_pixels

            total_pixels = reshaped_gt.shape[1] * reshaped_gt.shape[2] * reshaped_gt.shape[3]

            weight = num_pixels.to(diff.dtype) / total_pixels

            loss1 = (loss1 * weight).sum()

            loss += (loss1)

        return loss


def variance_depth_flow(gt_depth, patch_size, max_depth):
    batch_size, channels, height, width = gt_depth.size()

    patches_x = width // patch_size
    patches_y = height // patch_size

    gt_depth_patches = gt_depth.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    gt_depth_patches = gt_depth_patches.contiguous().view(batch_size, channels, patches_y, patches_x, patch_size, patch_size)
    gt_depth_patches = torch.where((gt_depth_patches > 0.1) * (gt_depth_patches < max_depth), gt_depth_patches, torch.zeros_like(gt_depth_patches))

    borders_north = torch.log(gt_depth_patches[:, :, :, :, 0, :]+1e-6) #1 n,c,30,40,16

    borders_south = torch.log(gt_depth_patches[:, :, :, :, -1, :]+1e-6) #3 n,c,30,40,16

    borders_west = torch.log(gt_depth_patches[:, :, :, :, :, 0]+1e-6) #0 n,c,30,40,16

    borders_east = torch.log(gt_depth_patches[:, :, :, :, :, -1]+1e-6) #2 - n,c,30,40,16
#
#     1
#   0   2
#     3
#

    # 0 i(2) - i+1 (0) - C
    diff_0 = (borders_east - torch.cat([borders_west[:,:,:,1:,:], torch.zeros_like(borders_west[:,:,:,0:1,:])], dim = 3)).sum(dim=-1, keepdim=True)
    # 1 i(3) - i+w (1) - C
    diff_1 = (borders_south - torch.cat([borders_north[:,:,1:,:,:], torch.zeros_like(borders_north[:,:,0:1,:,:])], dim = 2)).sum(dim=-1, keepdim=True)
    # 2 i(2) - i+1 (2) - C
    diff_2 = (borders_east - torch.cat([borders_east[:,:,:,1:,:], torch.zeros_like(borders_east[:,:,:,0:1,:])], dim=3)).sum(dim=-1, keepdim=True)
    # 3 i(3) - i+w (3) - C
    diff_3 = (borders_south - torch.cat([borders_south[:,:,1:,:,:], torch.zeros_like(borders_south[:,:,0:1,:,:]) ], dim=2)).sum(dim=-1, keepdim=True)
    # 4 i(0) - i+1 (0) - C
    diff_4 = (borders_west - torch.cat([borders_west[:,:,:,1:,:], torch.zeros_like(borders_west[:,:,:,0:1,:])],dim=3)).sum(dim=-1, keepdim=True)
    #5 i(1) - i+w (1) - C
    diff_5 = (borders_north - torch.cat([borders_north[:,:,1:,:,:], torch.zeros_like(borders_north[:,:,0:1,:,:])], dim=2)).sum(dim=-1, keepdim=True)
    #6 i(3) - i+1 (3) - C
    diff_6 = (borders_south - torch.cat([borders_south[:,:,:,1:,:], torch.zeros_like(borders_south[:,:,:,0:1,:])], dim=3)).sum(dim=-1, keepdim=True)
    #7 i(2) - i+w (2) - C
    diff_7 = (borders_east - torch.cat([borders_east[:,:,1:,:,:], torch.zeros_like(borders_east[:,:,0:1,:,:])], dim=2)).sum(dim=-1, keepdim=True)
    #8 i(1) - i+1 (1) - C
    diff_8 = (borders_north - torch.cat([borders_north[:,:,:,1:,:], torch.zeros_like(borders_north[:,:,:,0:1,:])], dim=3)).sum(dim=-1, keepdim=True)
    #9 i(0) - i+w (0) - C
    diff_9 = (borders_west - torch.cat([borders_west[:,:,1:,:,:], torch.zeros_like(borders_west[:,:,0:1,:,:])], dim=2)).sum(dim=-1, keepdim=True)

    diff = torch.cat([diff_0, diff_1, diff_2, diff_3, diff_4, diff_5, diff_6, diff_7, diff_8, diff_9], dim=-1).squeeze(1).permute(0,3,1,2).contiguous()
    diff = diff.reshape(batch_size, -1, patches_y, patches_x)/ patch_size

    return diff


class VarFlowLoss(nn.Module):
    def __init__(self, depth_channel, feat_channel, max_depth):
        super(VarFlowLoss, self).__init__()

        self.att = nn.Sequential(
                nn.Conv2d(feat_channel, depth_channel, kernel_size=3, padding=1),
                nn.Sigmoid())

        self.post = nn.Conv2d(depth_channel, 10, kernel_size=3, padding=1)
        self.max_depth = max_depth

    # feat, depth, ground_truth
    def forward(self, feat, depth, gts):
        os_shape = gts.shape[2:]
        n, c, h, w = depth.shape
        patch_size = (os_shape[0] // h)

        flow_gradient = variance_depth_flow(gts, patch_size, self.max_depth)

        att = self.att(feat)
        ds = self.post(att * depth) #match flow gradient
        diff = F.smooth_l1_loss(ds, flow_gradient, reduce=False, beta=0.01)
        return diff.sum()/(h*w*ds.shape[0]*ds.shape[1])
