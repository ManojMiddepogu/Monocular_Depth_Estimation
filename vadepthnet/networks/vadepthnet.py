import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .loss import VarLoss, SILogLoss, VarFlowLoss, ImprovedVarLoss
########################################################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=4),
            #nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            #ModulatedDeformConvPack(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )

        self.bt = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        skip = self.bt(x)

        x = self.channel_shuffle(x, 4)

        x = self.conv1(x)

        x = self.conv2(x)

        return x + skip

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape

        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        # (bs, channels, group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten (take all 1st among groups, 2nd among all groups....)
        x = x.view(batchsize, -1, height, width)

        return x



# upsample x1 and concat x2
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # doubt : why not transposed convolution
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(
            in_channels, out_channels, in_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX > 0 or diffY > 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class VarLayer(nn.Module):
    def __init__(self, in_channels, h, w, groups=4):
        super(VarLayer, self).__init__()

        self.gr = groups

        self.grad = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, 4*self.gr, kernel_size=3, padding=1))

        self.att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, 4*self.gr, kernel_size=3, padding=1),
                nn.Sigmoid())


        num = h * w

        a = torch.zeros(num, 4, num, dtype=torch.float16)

        for i in range(num):

            #a[i, 0, i] = 1.0
            #if i + 1 < num:
            # set x-direction differences
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 0, i] = 1.0
                a[i, 0, i+1] = -1.0

            #a[i, 1, i] = 1.0
            # set y-direction differences
            if i + w < num:
                a[i, 1, i] = 1.0
                a[i, 1, i+w] = -1.0

            if (i+2) % w != 0 and (i+1) % w !=0 and (i+2) < num:
                a[i, 2, i] = 1.0
                a[i, 2, i+2] = -1.0

            if i + w + w < num:
                a[i, 3, i] = 1.0
                a[i, 3, i+w+w] = -1.0

        a[-1, 0, -1] = 1.0
        a[-1, 1, -1] = 1.0

        a[-1, 2, -1] = 1.0
        a[-1, 3, -1] = 1.0

        self.register_buffer('a', a.unsqueeze(0)) # (1, hw, 4, hw)

        self.ins = nn.GroupNorm(1, self.gr)

        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels//2, self.gr, kernel_size=1, padding=0),
                nn.Sigmoid())

        self.post = nn.Sequential(
                # nn.Conv2d(self.gr, 8*self.gr, kernel_size=3, padding=1))
                nn.Conv2d(self.gr, 128, kernel_size=3, padding=1))

    def forward(self, x):
        att = self.att(x) # n, 4gr, h, w
        grad = self.grad(x) # n, 4gr, h, w

        se = self.se(x)

        n, c, h, w = x.shape

        att = att.reshape(n*self.gr, 4, h*w, 1).permute(0, 2, 1, 3) # (n*gr, hw, 4, 1)
        grad = grad.reshape(n*self.gr, 4, h*w, 1).permute(0, 2, 1, 3) # (n*gr, hw, 4, 1)

        x_processed = []
        loop_counter = min(8, att.shape[0] % 8)
        if loop_counter == 0:
            loop_counter = 8
        for i in range(0, att.shape[0], loop_counter):
            att_i = att[i:i+loop_counter]
            grad_i = grad[i:i+loop_counter]

            A_i = self.a * att_i # (loop_counter, hw, self.eq, hw)
            B_i = grad_i * att_i # (loop_counter, hw, self.eq, 1)
            A_i = A_i.reshape(loop_counter, h*w*4, h*w)
            B_i = B_i.reshape(loop_counter, h*w*4, 1)
            AT_i = A_i.permute(0, 2, 1)
            ATA_i = AT_i @ A_i
            del A_i
            ATB_i = AT_i @ B_i
            del B_i

            jitter = torch.eye(n=h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
            x_i = torch.linalg.solve(ATA_i + jitter, ATB_i)
            x_processed.append(x_i)
        x = torch.stack(x_processed, dim = 0)

        # # self.a = (hw, 4, hw)
        # A = self.a * att # (n*gr, hw, 4, hw)
        # B = grad * att # (n*gr, hw, 4, 1)

        # A = A.reshape(n*self.gr, h*w*4, h*w)
        # B = B.reshape(n*self.gr, h*w*4, 1)
        
        # # solve Ax = B
        # AT = A.permute(0, 2, 1)

        # ATA = torch.bmm(AT, A)
        # ATB = torch.bmm(AT, B)

        # jitter = torch.eye(n=h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
        # x = torch.linalg.solve(ATA + jitter, ATB)

        x = x.reshape(n, self.gr, h, w)

        x = self.ins(x) # group norm over x

        x = se * x # why?

        x = self.post(x) # (n, 8*gr, h, w)

        return x


class ImprovedVarLayer(nn.Module):
    def __init__(self, in_channels, h, w, groups=4):
        super(ImprovedVarLayer, self).__init__()

        self.channels_count = 6

        self.gr = groups
        self.grad = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, self.channels_count*self.gr, kernel_size=3, padding=1))
        self.att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, self.channels_count*self.gr, kernel_size=3, padding=1),
                nn.Sigmoid())


        num = h * w

        a = torch.zeros(num, self.channels_count, num, dtype=torch.float16)

        for i in range(num):
            # set x-direction differences
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 0, i] = 1.0
                a[i, 0, i+1] = -1.0

            # set y-direction differences
            if i + w < num:
                a[i, 1, i] = 1.0
                a[i, 1, i+w] = -1.0

            # set x-direction hop differences
            if (i+2) % w != 0 and (i+1) % w !=0 and (i+2) < num:
                a[i, 2, i] = 1.0
                a[i, 2, i+2] = -1.0

            # set y-direction hop differences
            if i + w + w < num:
                a[i, 3, i] = 1.0
                a[i, 3, i+w+w] = -1.0
            
            # set South West differences
            if (i) % w != 0 and (i+w) < num:
                a[i, 0, i] = 1.0
                a[i, 0, i+w-1] = -1.0
            
            # set South East differences
            if (i+1) % w != 0 and (i+w) < num:
                a[i, 0, i] = 1.0
                a[i, 0, i+w+1] = -1.0

        a[-1, 0, -1] = 1.0
        a[-1, 1, -1] = 1.0
        a[-1, 2, -1] = 1.0
        a[-1, 3, -1] = 1.0
        a[-1, 4, -1] = 1.0
        a[-1, 5, -1] = 1.0

        self.register_buffer('a', a.unsqueeze(0)) # (1, hw, 4, hw)

        self.ins = nn.GroupNorm(1, self.gr)

        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels//2, self.gr, kernel_size=1, padding=0),
                nn.Sigmoid())

        self.post = nn.Sequential(
                # nn.Conv2d(self.gr, 8*self.gr, kernel_size=3, padding=1))
                nn.Conv2d(self.gr, 128, kernel_size=3, padding=1))

    def forward(self, x):
        att = self.att(x) # n, 4gr, h, w
        grad = self.grad(x) # n, 4gr, h, w

        se = self.se(x)

        n, c, h, w = x.shape

        att = att.reshape(n*self.gr, self.channels_count, h*w, 1).permute(0, 2, 1, 3) # (n*gr, hw, 4, 1)
        grad = grad.reshape(n*self.gr, self.channels_count, h*w, 1).permute(0, 2, 1, 3) # (n*gr, hw, 4, 1)

        # x_processed = []
        # loop_counter = min(8, att.shape[0] % 8)
        # if loop_counter == 0:
        #     loop_counter = 8
        # for i in range(0, att.shape[0], loop_counter):
        #     att_i = att[i:i+loop_counter]
        #     grad_i = grad[i:i+loop_counter]

        #     A_i = self.a * att_i # (loop_counter, hw, self.eq, hw)
        #     B_i = grad_i * att_i # (loop_counter, hw, self.eq, 1)
        #     A_i = A_i.reshape(loop_counter, h*w*self.channels_count, h*w)
        #     B_i = B_i.reshape(loop_counter, h*w*self.channels_count, 1)
        #     AT_i = A_i.permute(0, 2, 1)
        #     ATA_i = AT_i @ A_i
        #     del A_i
        #     ATB_i = AT_i @ B_i
        #     del B_i

        #     jitter = torch.eye(n=h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
        #     x_i = torch.linalg.solve(ATA_i + jitter, ATB_i)
        #     x_processed.append(x_i)
        # x = torch.stack(x_processed, dim = 0)

        # self.a = (hw, 4, hw)
        A = self.a * att # (n*gr, hw, 4, hw)
        B = grad * att # (n*gr, hw, 4, 1)

        A = A.reshape(n*self.gr, h*w*self.channels_count, h*w)
        B = B.reshape(n*self.gr, h*w*self.channels_count, 1)
        # solve Ax = B
        AT = A.permute(0, 2, 1)

        jitter = torch.eye(n=h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
        x = torch.linalg.solve(torch.bmm(AT, A) + jitter, torch.bmm(AT, B))
        del A, AT, B

        x = x.reshape(n, self.gr, h, w)

        x = self.ins(x) # group norm over x

        x = se * x # why?

        x = self.post(x) # (n, 8*gr, h, w)

        return x

class FlowLayer(nn.Module):
    def __init__(self, in_channels, h, w, groups=4):
        super(FlowLayer, self).__init__()

        self.gr = groups
        self.eqn = 10

        self.grad = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, self.eqn*self.gr, kernel_size=3, padding=1))

        self.att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, self.eqn*self.gr, kernel_size=3, padding=1),
                nn.Sigmoid())

        num = h * w
        a = torch.zeros(num, self.eqn, 4 * num, dtype=torch.float16)

        for i in range(num):
            # set x-direction differences i(2) - i+1(0)
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 0, (2*num)+i] = 1.0
                a[i, 0, i+1] = -1.0
            # set y-direction differences i(3) - i+w (1)
            if i + w < num:
                a[i, 1, (3*num)+i] = 1.0
                a[i, 1, (num)+i+w] = -1.0
            # set x-direction differences i(2) - i+1(2)
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 2, (2*num)+i] = 1.0
                a[i, 2, (2*num)+i+1] = -1.0
            # set y-direction differences i(3) - i+w (3)
            if i + w < num:
                a[i, 3, (3*num)+i] = 1.0
                a[i, 3, (3*num)+i+w] = -1.0
            # set x-direction differences i(0) - i+1(0)
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 4, i] = 1.0
                a[i, 4, i+1] = -1.0
            # set y-direction differences i(1) - i+w (1)
            if i + w < num:
                a[i, 5, (num)+i] = 1.0
                a[i, 5, (num)+i+w] = -1.0
            # set x-direction differences i(3) - i+1(3)
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 6, (3*num)+i] = 1.0
                a[i, 6, (3*num)+i+1] = -1.0
            # set y-direction differences i(2) - i+w (2)
            if i + w < num:
                a[i, 7, (2*num)+i] = 1.0
                a[i, 7, (2*num)+i+w] = -1.0
            # set x-direction differences i(1) - i+1(1)
            if (i+1) % w != 0 and (i+1) < num:
                a[i, 8, num+i] = 1.0
                a[i, 8, num+i+1] = -1.0
            # set y-direction differences i(0) - i+w (0)
            if i + w < num:
                a[i, 9, i] = 1.0
                a[i, 9, i+w] = -1.0

        a[-1, 0, 2*num + num - 1] = 1.0 # 0
        a[-1, 1, 3*num + num - 1] = 1.0 # 1
        a[-1, 2, 2*num + num - 1] = 1.0 # 2
        a[-1, 3, 3*num + num - 1] = 1.0 # 3
        a[-1, 4, 0*num + num - 1] = 1.0 # 4
        a[-1, 5, 1*num + num - 1] = 1.0 # 5
        a[-1, 6, 3*num + num - 1] = 1.0 # 6
        a[-1, 7, 2*num + num - 1] = 1.0 # 7
        a[-1, 8, 1*num + num - 1] = 1.0 # 8
        a[-1, 9, 0*num + num - 1] = 1.0 # 9

        self.register_buffer('a', a.unsqueeze(0)) # (1, hw, self.eq, hw)

        self.ins = nn.GroupNorm(1, 4*self.gr)

        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
                nn.LeakyReLU(),
                # nn.Conv2d(in_channels//2, self.gr, kernel_size=1, padding=0),
                nn.Conv2d(in_channels//2, 4*self.gr, kernel_size=1, padding=0),
                nn.Sigmoid())

        self.post = nn.Sequential(
                # nn.Conv2d(4*self.gr, 4*8*self.gr, kernel_size=3, padding=1))
                nn.Conv2d(4*self.gr, 128, kernel_size=3, padding=1))

    def forward(self, x):
        # print_memory_usage("start")
        att = self.att(x) # n, 4*self.eq*gr, h, w
        # print_memory_usage("att")
        grad = self.grad(x) # n, 4*self.eq*gr, h, w
        # print_memory_usage("grad")

        se = self.se(x)

        n, c, h, w = x.shape

        att = att.reshape(n*self.gr, self.eqn, h*w, 1).permute(0, 2, 1, 3) # (n*gr, hw, self.eq, 1)
        # print_memory_usage("att_reshape")
        grad = grad.reshape(n*self.gr, self.eqn, h*w, 1).permute(0, 2, 1, 3) # (n*gr, hw, self.eq, 1)
        # print_memory_usage("grad_reshape")

        x_processed = []
        loop_counter = min(8, att.shape[0] % 8)
        if loop_counter == 0:
            loop_counter = 8
        # print(att.shape)
        for i in range(0, att.shape[0], loop_counter):
            att_i = att[i:i+loop_counter]
            # print_memory_usage(f"att_{i}")
            grad_i = grad[i:i+loop_counter]
            # print_memory_usage(f"grad_{i}")

            # print(i)
            # print(self.a.shape)
            # print(att_i.shape)
            A_i = self.a * att_i # (loop_counter, hw, self.eq, 4hw)
            # print_memory_usage(f"A_{i}")
            B_i = grad_i * att_i # (loop_counter, hw, self.eq, 1)
            # print_memory_usage(f"B_{i}")

            A_i = A_i.reshape(loop_counter, h*w*self.eqn, 4*h*w)
            B_i = B_i.reshape(loop_counter, h*w*self.eqn, 1)

            AT_i = A_i.permute(0, 2, 1)
            ATA_i = AT_i @ A_i
            del A_i
            ATB_i = AT_i @ B_i
            del B_i

            jitter = torch.eye(n=4*h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
            x_i = torch.linalg.solve(ATA_i + jitter, ATB_i)

            # # print_memory_usage(f"x_{i}_before")
            # x_i = torch.linalg.lstsq(A_i, B_i)
            # # print_memory_usage(f"x_{i}_after")
            # del A_i
            # # print_memory_usage(f"A_{i}_deleted")
            # del B_i
            # # print_memory_usage(f"B_{i}_deleted")
            # x_i = x_i.solution
            # # print_memory_usage(f"x_{i}")

            x_processed.append(x_i)

        x = torch.stack(x_processed, dim = 0)
        # print("x processed shape: ", x.shape)

        # # self.a = (hw, self.eq, 4hw)
        # A = self.a * att # (n*gr, hw, self.eq, 4hw)
        # B = grad * att # (n*gr, hw, self.eq, 1)

        # A = A.reshape(n*self.gr, h*w*self.eqn, 4*h*w)
        # B = B.reshape(n*self.gr, h*w*self.eqn, 1)
        
        # # solve Ax = B
        # # X Shape -> (n*gr, 4hw, 1)
        # AT = A.permute(0, 2, 1)

        # ATA = torch.bmm(AT, A)
        # ATB = torch.bmm(AT, B)

        # jitter = torch.eye(n=4*h*w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
        # x = torch.linalg.solve(ATA + jitter, ATB)
        # # x = torch.linalg.lstsq(A, B)
        # # del A
        # # del B
        # # x = x.solution

        # x -> (n*gr, 4hw, 1)
        x = x.reshape(n, 4*self.gr, h, w)
        # x -> (n, 4*gr, h, w)

        x = self.ins(x) # group norm over x

        x = se * x # why?

        x = self.post(x) # (n, 8*gr, h, w)

        return x


class Refine(nn.Module):
    def __init__(self, c1, c2):
        super(Refine, self).__init__()

        s = c1 + c2
        # features, depth
        self.fw = nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(s, c1, kernel_size=3, padding=1))

        self.dw = nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(s, c2, kernel_size=3, padding=1))

    def forward(self, feat, depth):
        cc = torch.cat([feat, depth], 1)
        feat_new = self.fw(cc)
        depth_new = self.dw(cc)
        return feat_new, depth_new


class MetricLayer(nn.Module):
    def __init__(self, c):
        super(MetricLayer, self).__init__()

        self.ln = nn.Sequential(
                nn.Linear(c, c//4),
                nn.LeakyReLU(),
                nn.Linear(c//4, 2))
        # s, t

    def forward(self, x):

        x = x.squeeze(-1).squeeze(-1)
        x = self.ln(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x


class VADepthNet(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640), swin_type = "tiny"):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth

        self.swin_type = swin_type

        pretrain_img_size = img_size
        patch_size = (4, 4)
        in_chans = 3

        if self.swin_type == "large":
            # large size swin
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            window_size = 12
        if self.swin_type == "small":
            # small size swin
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7
        if self.swin_type == "tiny":
            # tiny size swin
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )


        self.backbone = SwinTransformer(**backbone_cfg)
        
        self.backbone.init_weights(pretrained=pretrained)

        if self.swin_type == "large":
            # large swin model
            self.up_4 = Up(1536 + 768, 512)
            self.up_3 = Up(512 + 384, 256)
            self.up_2 = Up(256 + 192, 64)
        if self.swin_type == "small":
            # small swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)
        if self.swin_type == "tiny":
            # tiny swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)

        self.outc = OutConv(128, 1, self.prior_mean) # change this?

        self.vlayer = VarLayer(512, img_size[0]//16, img_size[1]//16, groups=16)

        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        self.var_loss = VarLoss(128, 512)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        if self.swin_type == "large":
            # large swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536))
        if self.swin_type == "small":
            # small swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))
        if self.swin_type == "tiny":
            # tiny swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))

    def forward(self, x, gts=None):

        x2, x3, x4, x5 = self.backbone(x)
        # torch.Size([8, 192, 120, 160]) torch.Size([8, 384, 60, 80]) torch.Size([8, 768, 30, 40]) torch.Size([8, 1536, 15, 20])
        # print(x2.shape, x3.shape, x4.shape, x5.shape)

        outs = {}

        metric = self.mlayer(x5) # (n, 2)

        x = self.up_4(x5, x4) # (scale 2*x5, x4)
        
        d = self.vlayer(x)

        if self.training:
            # print("training print: ", x.shape, d.shape, gts.shape)
            var_loss = self.var_loss(x, d, gts)


        x, d  = self.ref_4(x, d) # refine feat and depth, no size change

        d_u4 = F.interpolate(d, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.up_3(x, x3)

        # double d and pass
        x, d = self.ref_3(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))

        d_u3 = F.interpolate(d, scale_factor=8, mode='bilinear', align_corners=True)

        x = self.up_2(x, x2)

        x, d = self.ref_2(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))

        d_u2 = F.interpolate(d, scale_factor=4, mode='bilinear', align_corners=True)

        d = d_u2 + d_u3 + d_u4 # change this?

        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))

        outs['scale_1'] = d

        if self.training:
            si_loss = self.si_loss(outs, gts)
            loss_dict = {
                "var_loss": var_loss,
                "si_loss": si_loss
            }
            return outs['scale_1'], loss_dict
        else:
            return outs['scale_1']


class ImprovedVADepthNet(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640), swin_type = "tiny"):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth
        self.swin_type = swin_type

        pretrain_img_size = img_size
        patch_size = (4, 4)
        in_chans = 3

        if self.swin_type == "large":
            # large size swin
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            window_size = 12
        if self.swin_type == "small":
            # small size swin
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7
        if self.swin_type == "tiny":
            # tiny size swin
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )

        self.backbone = SwinTransformer(**backbone_cfg)        
        self.backbone.init_weights(pretrained=pretrained)

        if self.swin_type == "large":
            # large swin model
            self.up_4 = Up(1536 + 768, 512)
            self.up_3 = Up(512 + 384, 256)
            self.up_2 = Up(256 + 192, 64)
        if self.swin_type == "small":
            # small swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)
        if self.swin_type == "tiny":
            # tiny swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)

        self.outc = OutConv(128, 1, self.prior_mean) # change this?

        if self.swin_type == "large":
            self.vlayer32 = VarLayer(1536, img_size[0]//32, img_size[1]//32, groups=16)
        else:
            self.vlayer32 = ImprovedVarLayer(1536//2, img_size[0]//32, img_size[1]//32, groups=16)
        self.vlayer16 = ImprovedVarLayer(512, img_size[0]//16, img_size[1]//16, groups=16)
        # self.vlayer8 = ImprovedVarLayer(256, img_size[0]//8, img_size[1]//8, groups=4)
        # self.vlayer4 = ImprovedVarLayer(64, img_size[0]//4, img_size[1]//4, groups=4)

        if self.swin_type == "large":
            self.ref_5 = Refine(1536, 128)
        else:
            self.ref_5 = Refine(1536//2, 128)
        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        if self.swin_type == "large":
            self.var_loss_5 = ImprovedVarLoss(128, 1536)
        else:
            self.var_loss_5 = ImprovedVarLoss(128, 1536//2)
        self.var_loss_4 = ImprovedVarLoss(128, 512)
        # self.var_loss_3 = ImprovedVarLoss(128, 256)
        # self.var_loss_2 = ImprovedVarLoss(128, 64)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        if self.swin_type == "large":
            # large swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536))
        if self.swin_type == "small":
            # small swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))
        if self.swin_type == "tiny":
            # tiny swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))

    def forward(self, x, gts=None):
        var_loss = 0.0
        x2, x3, x4, x5 = self.backbone(x)

        x = x5

        outs = {}
        metric = self.mlayer(x) # (n, 2)

        d32 = self.vlayer32(x)
        if self.training:
            var_loss += self.var_loss_5(x, d32, gts)
        
        x, d  = self.ref_5(x, d32)
        d_u5 = F.interpolate(d, scale_factor=32, mode='bilinear', align_corners=True)

        x = self.up_4(x, x4)

        d16 = self.vlayer16(x)
        if self.training:
            var_loss += self.var_loss_4(x, d16, gts)

        x, d  = self.ref_4(x, d16) # refine feat and depth, no size change
        d_u4 = F.interpolate(d, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.up_3(x, x3)

        # d8 = self.vlayer8(x)
        # if self.training:
        #     var_loss += self.var_loss_3(x, d8, gts)

        # x, d = self.ref_3(x, d8)
        x, d8 = self.ref_3(x, F.interpolate(d16, scale_factor=2, mode='bilinear', align_corners=True))
        d_u3 = F.interpolate(d8, scale_factor=8, mode='bilinear', align_corners=True)

        x = self.up_2(x, x2)

        # d4 = self.vlayer4(x)
        # if self.training:
        #     var_loss += self.var_loss_2(x, d4, gts)

        # x, d = self.ref_2(x, d4)
        x, d4 = self.ref_2(x, F.interpolate(d8, scale_factor=2, mode='bilinear', align_corners=True))
        d_u2 = F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True)

        d = d_u2 + d_u3 + d_u4 + d_u5 # change this?
        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))

        outs['scale_1'] = d

        if self.training:
            si_loss = self.si_loss(outs, gts)
            loss_dict = {
                "var_loss": var_loss,
                "si_loss": si_loss
            }
            return outs['scale_1'], loss_dict
        else:
            return outs['scale_1']


class SelfAttention(nn.Module):
    def __init__(self, in_channels, scale):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, max(1, in_channels // scale), kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, max(1, in_channels // scale), kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, max(1, in_channels // scale), kernel_size=1)
        self.scale = scale

    def forward(self, x):
        batches, channels, h, w = x.shape
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = torch.matmul(proj_query.view(proj_query.size(0), -1, proj_query.size(2)*proj_query.size(3)),
                              proj_key.view(proj_key.size(0), -1, proj_key.size(2)*proj_key.size(3)).permute(0, 2, 1))
        attention = F.softmax(energy, dim=1)

        out = torch.matmul(attention, proj_value.view(proj_value.size(0), -1, proj_value.size(2)*proj_value.size(3)))
        out = out.view(batches, channels // self.scale, h, w)
        return out

class DepthImageCombination(nn.Module):
    def __init__(self, num_channels):
        super(DepthImageCombination, self).__init__()
        self.self_attention = SelfAttention(num_channels, 8)

    def forward(self, depth_images):
        combined = self.self_attention(depth_images)
        return combined


class VAFlowNet(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640), swin_type="tiny", depth_resnet_connection=False, use_self_attention=False):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth
        self.swin_type = swin_type
        self.use_self_attention = use_self_attention
        self.depth_resnet_connection = depth_resnet_connection

        pretrain_img_size = img_size
        patch_size = (4, 4)
        in_chans = 3

        if self.swin_type == "large":
            # large size swin
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            window_size = 12
        if self.swin_type == "small":
            # small size swin
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7
        if self.swin_type == "tiny":
            # tiny size swin
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        
        self.backbone.init_weights(pretrained=pretrained)

        if self.swin_type == "large":
            # large swin model
            self.up_4 = Up(1536 + 768, 512)
            self.up_3 = Up(512 + 384, 256)
            self.up_2 = Up(256 + 192, 64)
        if self.swin_type == "small":
            # small swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)
        if self.swin_type == "tiny":
            # tiny swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)

        if self.depth_resnet_connection and self.use_self_attention:
            self.outc = OutConv(32, 1, self.prior_mean) # change this?
        else:
            self.outc = OutConv(128, 1, self.prior_mean) # change this?

        if self.depth_resnet_connection:
            self.vlayer = VarLayer(512, img_size[0]//16, img_size[1]//16, groups=4)

        self.flayer16 = FlowLayer(512, img_size[0]//16, img_size[1]//16, groups=4)
        # self.flayer8 = FlowLayer(256, img_size[0]//8, img_size[1]//8, groups=4)
        # self.flayer4 = FlowLayer(64, img_size[0]//4, img_size[1]//4, groups=4)

        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        if self.depth_resnet_connection:
            self.ref_4_vlayer = Refine(512, 128)
            self.ref_3_vlayer = Refine(256, 128)
            self.ref_2_vlayer = Refine(64, 128)

        if self.use_self_attention:
            self.depth_combination = DepthImageCombination(128*2)
        if self.depth_resnet_connection:            
            self.var_loss = VarLoss(128, 512)
        self.var_flow_loss = VarFlowLoss(128, 512, self.max_depth)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        if self.swin_type == "large":
            # large swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536))
        if self.swin_type == "small":
            # small swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))
        if self.swin_type == "tiny":
            # tiny swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))

    def forward(self, x, gts=None):
        # print_memory_usage("REAL START")
        # print("x.shape start: ", x.shape)
        x2, x3, x4, x5 = self.backbone(x)
        # print_memory_usage("BACKBONE DONE")
        # torch.Size([8, 192, 120, 160]) torch.Size([8, 384, 60, 80]) torch.Size([8, 768, 30, 40]) torch.Size([8, 1536, 15, 20])
        # print(x2.shape, x3.shape, x4.shape, x5.shape)

        outs = {}

        metric = self.mlayer(x5) # (n, 2)
        # print_memory_usage("MLAYER DONE")

        x = self.up_4(x5, x4) # (scale 2*x5, x4)
        # print_memory_usage("UP4 DONE")
        # print("x.shape before flayer: ", x.shape)
        f16 = self.flayer16(x)
        if self.depth_resnet_connection:
            d16_vlayer = self.vlayer(x)

        if self.training:
            var_flow_loss = self.var_flow_loss(x, f16, gts)
            if self.depth_resnet_connection:
                var_loss = self.var_loss(x, d16_vlayer, gts)

        x_flow, d16  = self.ref_4(x, f16) # refine feat and depth, no size change
        if self.depth_resnet_connection:
            x_vlayer, d16_vlayer  = self.ref_4_vlayer(x, d16_vlayer) # refine feat and depth, no size change

        d_u4 = F.interpolate(d16, scale_factor=16, mode='bilinear', align_corners=True)
        if self.depth_resnet_connection:
            d_u4_vlayer = F.interpolate(d16_vlayer, scale_factor=16, mode='bilinear', align_corners=True)

        x_flow = self.up_3(x_flow, x3)
        if self.depth_resnet_connection:
            x_vlayer = self.up_3(x_vlayer, x3)

        # f8 = self.flayer8(x)

        # double d and pass
        x_flow, d8 = self.ref_3(x_flow, F.interpolate(d16, scale_factor=2, mode='bilinear', align_corners=True))
        if self.depth_resnet_connection:
            x_vlayer, d8_vlayer = self.ref_3_vlayer(x_vlayer, F.interpolate(d16_vlayer, scale_factor=2, mode='bilinear', align_corners=True))
        # x, d8 = self.ref_3(x, f8)

        d_u3 = F.interpolate(d8, scale_factor=8, mode='bilinear', align_corners=True)
        if self.depth_resnet_connection:
            d_u3_vlayer = F.interpolate(d8_vlayer, scale_factor=8, mode='bilinear', align_corners=True)

        x_flow = self.up_2(x_flow, x2)
        if self.depth_resnet_connection:
            x_vlayer = self.up_2(x_vlayer, x2)

        # f4 = self.flayer4(x)

        x_flow, d4 = self.ref_2(x_flow, F.interpolate(d8, scale_factor=2, mode='bilinear', align_corners=True))
        if self.depth_resnet_connection:
            x_vlayer, d4_vlayer = self.ref_2_vlayer(x_vlayer, F.interpolate(d8_vlayer, scale_factor=2, mode='bilinear', align_corners=True))
        # x, d4 = self.ref_2(x, f4)

        d_u2 = F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True)
        if self.depth_resnet_connection:
            d_u2_vlayer = F.interpolate(d4_vlayer, scale_factor=4, mode='bilinear', align_corners=True)

        d = d_u2 + d_u3 + d_u4 # change this?
        if self.depth_resnet_connection:
            d_vlayer = d_u2_vlayer + d_u3_vlayer + d_u4_vlayer # change this?

        if self.depth_resnet_connection:
            if self.use_self_attention:
                d = torch.cat([d,d_vlayer], dim=1)
                d = self.depth_combination(d)
            else:
                d = d + d_vlayer

        # print(d.shape)
        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))

        outs['scale_1'] = d

        if self.training:
            si_loss = self.si_loss(outs, gts)
            if self.depth_resnet_connection:
                loss_dict = {
                    "var_flow_loss": var_flow_loss,
                    "var_loss": var_loss,
                    "si_loss": si_loss
                }
            else:
                loss_dict = {
                    "var_flow_loss": var_flow_loss,
                    "si_loss": si_loss
                }
            return outs['scale_1'], loss_dict
        else:
            return outs['scale_1']



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, prior_mean = 1.54):
        super(OutConv, self).__init__()

        self.prior_mean = prior_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.exp(self.conv(x) + self.prior_mean)


class VAFlowNet32(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640), swin_type="tiny", use_self_attention=False):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth
        self.swin_type = swin_type
        self.use_self_attention = use_self_attention

        pretrain_img_size = img_size
        patch_size = (4, 4)
        in_chans = 3

        if self.swin_type == "large":
            # large size swin
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            window_size = 12
        if self.swin_type == "small":
            # small size swin
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7
        if self.swin_type == "tiny":
            # tiny size swin
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            window_size = 7

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        
        self.backbone.init_weights(pretrained=pretrained)

        if self.swin_type == "large":
            # large swin model
            self.up_4 = Up(1536 + 768, 512)
            self.up_3 = Up(512 + 384, 256)
            self.up_2 = Up(256 + 192, 64)
        if self.swin_type == "small":
            # small swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)
        if self.swin_type == "tiny":
            # tiny swin model
            self.up_4 = Up((1536//2) + (768//2), 512)
            self.up_3 = Up(512 + (384//2), 256)
            self.up_2 = Up(256 + (192//2), 64)

        if self.use_self_attention:
            self.outc = OutConv(64, 1, self.prior_mean) # change this?
        else:
            self.outc = OutConv(128, 1, self.prior_mean) # change this?

        self.vlayer = VarLayer(512, img_size[0]//16, img_size[1]//16, groups=4)
        self.flayer32 = FlowLayer(1536//2, img_size[0]//32, img_size[1]//32, groups=4)

        self.ref_5 = Refine(1536//2, 128)
        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        if self.use_self_attention:
            self.depth_combination = DepthImageCombination(128*4)
        
        self.var_loss = VarLoss(128, 512)
        self.var_flow_loss = VarFlowLoss(128, 1536 // 2, self.max_depth)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        if self.swin_type == "large":
            # large swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536))
        if self.swin_type == "small":
            # small swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))
        if self.swin_type == "tiny":
            # tiny swin model
            self.mlayer = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1,1)),
                    MetricLayer(1536//2))

    def forward(self, x, gts=None):
        x2, x3, x4, x5 = self.backbone(x)
        outs = {}

        metric = self.mlayer(x5) # (n, 2)

        f32 = self.flayer32(x5) # (128, 15, 20)
        if self.training:
            var_flow_loss = self.var_flow_loss(x5, f32, gts)

        x_flow, d32 = self.ref_5(x5, f32) # refine feat and depth, no size change
        d_u5 = F.interpolate(d32, scale_factor=32, mode='bilinear', align_corners=True)

        x = self.up_4(x_flow, x4) # (scale 2*x5, x4)
        
        d16_vlayer = self.vlayer(x)
        if self.training:
            var_loss = self.var_loss(x, d16_vlayer, gts)

        x_vlayer, d16_vlayer  = self.ref_4(x, d16_vlayer) # refine feat and depth, no size change
        d_u4_vlayer = F.interpolate(d16_vlayer, scale_factor=16, mode='bilinear', align_corners=True)

        x_vlayer = self.up_3(x_vlayer, x3)

        # double d and pass
        x_vlayer, d8_vlayer = self.ref_3(x_vlayer, F.interpolate(d16_vlayer, scale_factor=2, mode='bilinear', align_corners=True))
        d_u3_vlayer = F.interpolate(d8_vlayer, scale_factor=8, mode='bilinear', align_corners=True)

        x_vlayer = self.up_2(x_vlayer, x2)

        x_vlayer, d4_vlayer = self.ref_2(x_vlayer, F.interpolate(d8_vlayer, scale_factor=2, mode='bilinear', align_corners=True))
        d_u2_vlayer = F.interpolate(d4_vlayer, scale_factor=4, mode='bilinear', align_corners=True)

        if self.use_self_attention:
            d = torch.cat([d_u5, d_u2_vlayer, d_u3_vlayer, d_u4_vlayer], dim=1)
            d = self.depth_combination(d)
        else:
            d = d_u5 + d_u2_vlayer + d_u3_vlayer + d_u4_vlayer # change this?

        # print(d.shape)
        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))

        outs['scale_1'] = d

        if self.training:
            si_loss = self.si_loss(outs, gts)
            loss_dict = {
                "var_flow_loss": var_flow_loss,
                "var_loss": var_loss,
                "si_loss": si_loss
            }
            return outs['scale_1'], loss_dict
        else:
            return outs['scale_1']


def print_memory_usage(message=""):
    """
    Print CUDA memory usage.

    Args:
    - message: Optional message to display.
    """
    memory_used = torch.cuda.memory_allocated() / (1024 ** 2)
    max_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"{message}Memory used: {memory_used:.2f} MB, Max memory used: {max_memory_used:.2f} MB")