import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from networks.vadepthnet import *
from dataloaders.dataloader import ToTensor

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

# SOTA
# model = VADepthNet(max_depth=10, prior_mean=1.54, img_size=(480, 640), swin_type="large")
# model = torch.nn.DataParallel(model).cuda()
# checkpoint = torch.load('/scratch/crg9968/cv/Monocular_Depth_Estimation/ckpts/vadepthnet_nyu.pth', map_location=device)
# model.load_state_dict(checkpoint['model'])
# image_name_save = "SOTA"
# model.eval()

# Baseline Tiny
# model = VADepthNet(max_depth=10, prior_mean=1.54, img_size=(480, 640), swin_type="tiny")
# model = torch.nn.DataParallel(model).cuda()
# checkpoint = torch.load('/scratch/crg9968/cv/Monocular_Depth_Estimation/vadepthnet/baseline_tiny/vadepthnet/model-24000-best_d2_0.98550', map_location=device)
# model.load_state_dict(checkpoint['model'])
# image_name_save = "baseline_tiny"
# model.eval()

# Baseline Tiny 4
# model = VADepthNet(max_depth=10, prior_mean=1.54, img_size=(480, 640), swin_type="tiny")
# model = torch.nn.DataParallel(model).cuda()
# checkpoint = torch.load('/scratch/crg9968/cv/Monocular_Depth_Estimation/vadepthnet/baseline_tiny_4/vadepthnet/model-13000-best_silog_10.74548', map_location=device)
# model.load_state_dict(checkpoint['model'])
# image_name_save = "baseline_tiny_4"
# model.eval()

# model = WindowVADepthNet(max_depth=10, prior_mean=1.54, img_size=(480, 640), swin_type="tiny", window_size_directions=3)
# model = torch.nn.DataParallel(model).cuda()
# checkpoint = torch.load('/scratch/crg9968/cv/Monocular_Depth_Estimation/vadepthnet/window_3_tiny/windowvadepthnet/model-23100-best_log10_0.04357', map_location=device)
# model.load_state_dict(checkpoint['model'])
# image_name_save = "window_3"
# model.eval()

# model = WindowVADepthNet(max_depth=10, prior_mean=1.54, img_size=(480, 640), swin_type="tiny", window_size_directions=2)
# model = torch.nn.DataParallel(model).cuda()
# checkpoint = torch.load('/scratch/crg9968/cv/Monocular_Depth_Estimation/vadepthnet/window_2_tiny/windowvadepthnet/model-8100-best_silog_10.28274', map_location=device)
# model.load_state_dict(checkpoint['model'])
# image_name_save = "window_2"
# model.eval()

model = VAFlowNet(max_depth=10, prior_mean=1.54, img_size=(480, 640), swin_type="tiny")
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('/scratch/crg9968/cv/Monocular_Depth_Estimation/vadepthnet/vaflownet_tiny/vaflownet/model-36000-best_silog_11.20024', map_location=device)
model.load_state_dict(checkpoint['model'])
image_name_save = "flow"
model.eval()

# # Load an example image
# image_path = '/scratch/crg9968/cv/Monocular_Depth_Estimation/dataset/nyu_depth_v2/official_splits/test/home_office/rgb_00360.jpg'
image_path = '/scratch/crg9968/cv/Monocular_Depth_Estimation/rgb.png'
original_img = Image.open(image_path)
print(original_img.size)
np_img = np.asarray(original_img, dtype=np.float32) / 255.0

# Preprocess the image
totensor = ToTensor('test')
img = totensor.to_tensor(np_img)
img = totensor.normalize(img)
img = img.unsqueeze(0).cuda()

# Forward pass through the model
pdepth = model.forward(img)

# Convert depth tensor to a numpy array and normalize
pdepth_np = pdepth.squeeze().cpu().detach().numpy()
pdepth_np = (pdepth_np - pdepth_np.min()) / (pdepth_np.max() - pdepth_np.min())  # Normalize
pdepth_np = 1.0 - pdepth_np

# Convert the depth map to a colormap
depth_colormap = plt.get_cmap('plasma')(pdepth_np)[:, :, :3]  # Exclude alpha channel
depth_colormap = (depth_colormap * 255).astype(np.uint8)

# Convert depth map and original image to the same size
depth_colormap = Image.fromarray(depth_colormap)
depth_colormap = depth_colormap.resize(original_img.size, Image.BILINEAR)

# Combine original image and depth map side by side
# combined_image = Image.new('RGB', (original_img.width + depth_colormap.width, original_img.height))
# combined_image.paste(original_img, (0, 0))
# combined_image.paste(depth_colormap, (original_img.width, 0))

combined_image = Image.new('RGB', (depth_colormap.width, original_img.height))
combined_image.paste(depth_colormap, (0, 0))


# Save the combined image
combined_image.save(f'{image_name_save}.png')

# depth_path = '/scratch/crg9968/cv/Monocular_Depth_Estimation/dataset/nyu_depth_v2/official_splits/test/living_room/sync_depth_00210.png'
depth_path = '/scratch/crg9968/cv/Monocular_Depth_Estimation/depth.png'
original_depth = Image.open(depth_path)

# depth_gt = np.array(original_depth)
# valid_mask = np.zeros_like(depth_gt)
# valid_mask[45:472, 43:608] = 1
# depth_gt[valid_mask==0] = 0
# original_depth = Image.fromarray(depth_gt)

# np_depth = np.asarray(original_depth, dtype=np.float32)
np_depth = np.asarray(original_depth, dtype=np.float32) / 1000.0
print(np_depth.max(), np_depth.min())

np_depth = (np_depth - np_depth.min()) / (np_depth.max() - np_depth.min())  # Normalize
np_depth = 1.0 - np_depth
# print(np_depth.max(), np_depth.min())

print(plt.get_cmap('plasma')(np_depth).shape)

# Convert the depth map to a colormap
depth_colormap = plt.get_cmap('plasma')(np_depth)[:, :, :3]  # Exclude alpha channel
depth_colormap = (depth_colormap * 255).astype(np.uint8)

print(depth_colormap.size)

# Convert depth map and original image to the same size
depth_colormap = Image.fromarray(depth_colormap)
print(depth_colormap.size)
print(original_img.size)
depth_colormap = depth_colormap.resize(original_img.size, Image.BILINEAR)

# Combine original image and depth map side by side
groundtruth_image = Image.new('RGB', (depth_colormap.width, depth_colormap.height))
groundtruth_image.paste(depth_colormap, (0, 0))

# Save the combined image
groundtruth_image.save('groundtruth_image.png')
