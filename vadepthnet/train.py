import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm
from dataloaders.dataloader import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

# from tensorboardX import SummaryWriter

from utils import compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
from networks.vadepthnet import VADepthNet, VAFlowNet, VAFlowNet32, ImprovedVADepthNet


parser = argparse.ArgumentParser(description='VADepthNet PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='vadepthnet')
parser.add_argument('--use_self_attention',                    help='Use self attention on depths', action='store_true')
parser.add_argument('--depth_resnet_connection',               help='Use vlayer and flayer', action='store_true')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--prior_mean',                type=float, help='prior mean of depth', default=1.54)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)
parser.add_argument('--use_wandb',                             help='Save logs in wandb', action='store_true')

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--vis_freq',                  type=int,   help='Visualization Frequency in global steps', default=100)
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader


def online_eval(model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(image)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    if args.use_wandb:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            wandb.login(key='1d3bfdab8059559c1286c808290e032fca67f654')
            wandb.init(
                entity = "nyu_chanukya",
                project="monocular-depth-estimation",
                config = args,
            )
    
    swin_type = args.pretrain.split("/")[-1].split("_")[1]
    if not swin_type in ["tiny", "small", "large"]:
        raise ValueError(f"Invalid swin model type {swin_type}!")

    print_memory_usage("MODEL NOT LOADED YET")
    if args.model_name == "vadepthnet":
        model = VADepthNet(pretrained=args.pretrain,
                        max_depth=args.max_depth,
                        prior_mean=args.prior_mean,
                        img_size=(args.input_height, args.input_width),
                        swin_type=swin_type)
    elif args.model_name == "improvedvadepthnet":
        model = ImprovedVADepthNet(pretrained=args.pretrain,
                        max_depth=args.max_depth,
                        prior_mean=args.prior_mean,
                        img_size=(args.input_height, args.input_width),
                        swin_type=swin_type)
    elif args.model_name == "vaflownet":
        model = VAFlowNet(pretrained=args.pretrain,
                        max_depth=args.max_depth,
                        prior_mean=args.prior_mean,
                        img_size=(args.input_height, args.input_width),
                        swin_type=swin_type,
                        depth_resnet_connection=args.depth_resnet_connection,
                        use_self_attention=args.use_self_attention)
    elif args.model_name == "vaflownet32":
        model = VAFlowNet32(pretrained=args.pretrain,
                        max_depth=args.max_depth,
                        prior_mean=args.prior_mean,
                        img_size=(args.input_height, args.input_width),
                        swin_type=swin_type,
                        use_self_attention=args.use_self_attention)
    else:
        raise ValueError(f"Invalid model name {args.model_name}")
    model.train()
    # print_memory_usage("MODEL LOADED HERE")

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    # print_memory_usage("MODEL BEFORE GPU HERE")
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    # print_memory_usage("MODEL TO GPU HERE")

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Evaluation before training ======
    # model.eval()
    # with torch.no_grad():
    #     eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, post_process=True)

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        # writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            # eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    #silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            # print("Image shape: ", image.shape)

            # print_memory_usage("MODEL BEFORE")
            depth_est, loss_dict = model(image, depth_gt)
            loss = sum(loss_dict.values())
            # print_memory_usage("MODEL AFTER")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            #optimizer.step()


            duration += time.time() - before_op_time
            if global_step % args.log_freq == 0 and gpu == 0:
                print('epoch:', epoch, 'global_step:', global_step, 'loss:', loss.item(), 'duration:', duration, flush=True)
                if args.use_wandb:
                    log_dict = {"train_loss": loss.item(), "epoch": epoch, "duration": duration}
                    for k, v in loss_dict.items():
                        log_dict[k] = v.item()
                    wandb.log(log_dict, step=int(global_step))
            
            # if global_step and global_step % args.vis_freq == 0 and gpu == 0:
            if global_step % args.vis_freq == 0 and gpu == 0:
                image_paths = [
                    '../dataset/nyu_depth_v2/sync/cafe_0001a/rgb_00021.jpg',
                    '../dataset/nyu_depth_v2/official_splits/test/living_room/rgb_00210.jpg',
                    '../dataset/nyu_depth_v2/official_splits/test/home_office/rgb_00360.jpg'
                ]

                totensor = ToTensor('test')
                original_images = []
                images = []
                for image_path in image_paths:
                    original_img = Image.open(image_path)
                    original_images.append(original_img)

                    img = np.asarray(original_img, dtype=np.float32) / 255.0
                    img = totensor.to_tensor(img)
                    img = totensor.normalize(img)
                    images.append(img.unsqueeze(0))

                batch_tensor = torch.cat(images, dim=0).cuda()

                # Model forward pass in batch
                model.eval()
                with torch.no_grad():
                    pdepth_batch = model(batch_tensor)
                model.train()

                combined_images = []
                for i, original_img in enumerate(original_images):
                    pdepth_np = pdepth_batch[i].squeeze().cpu().detach().numpy()
                    pdepth_np = (pdepth_np - pdepth_np.min()) / (pdepth_np.max() - pdepth_np.min())  # Normalize
                    pdepth_np = 1.0 - pdepth_np

                    depth_colormap = plt.get_cmap('plasma')(pdepth_np)[:, :, :3]
                    depth_colormap = (depth_colormap * 255).astype(np.uint8)
                    depth_colormap = Image.fromarray(depth_colormap).resize(original_img.size, Image.BILINEAR)

                    combined_image = Image.new('RGB', (original_img.width + depth_colormap.width, original_img.height))
                    combined_image.paste(original_img, (0, 0))
                    combined_image.paste(depth_colormap, (original_img.width, 0))

                    combined_images.append(combined_image)

                # Combine all images side by side
                total_width = sum(image.width for image in combined_images)
                max_height = max(image.height for image in combined_images)
                final_image = Image.new('RGB', (total_width, max_height))

                x_offset = 0
                for image in combined_images:
                    final_image.paste(image, (x_offset, 0))
                    x_offset += image.width

                # Log the final combined image to wandb
                if args.use_wandb:
                    wandb.log({"Combined Image": [wandb.Image(final_image, caption="Original and Depth Side by Side")]}, step=int(global_step))

            # if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, post_process=False)
                if eval_measures is not None:
                    for i in range(len(eval_metrics)):
                        # eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        if args.use_wandb:
                            wandb.log({eval_metrics[i]: eval_measures[i].cpu().item()}, step=int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    # eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1
       
    # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
    #     writer.close()
    #     if args.do_online_eval:
    #         eval_summary_writer.close()


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    log_directory_path = os.path.join(args.log_directory, args.model_name)
    os.makedirs(log_directory_path, exist_ok=True)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def print_memory_usage(message=""):
    """
    Print CUDA memory usage.

    Args:
    - message: Optional message to display.
    """
    memory_used = torch.cuda.memory_allocated() / (1024 ** 2)
    max_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"{message}Memory used: {memory_used:.2f} MB, Max memory used: {max_memory_used:.2f} MB")

if __name__ == '__main__':
    main()
