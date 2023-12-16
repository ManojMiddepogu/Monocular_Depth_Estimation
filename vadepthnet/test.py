import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from networks.vadepthnet import VADepthNet
from dataloaders.dataloader import ToTensor

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

model = VADepthNet(max_depth=10, prior_mean=1.54, img_size=(480, 640))
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('../ckpts/vadepthnet_nyu.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Load an example image
image_path = '../dataset/nyu_depth_v2/sync/kitchen_0028b/rgb_00045.jpg'
original_img = Image.open(image_path)
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
combined_image = Image.new('RGB', (original_img.width + depth_colormap.width, original_img.height))
combined_image.paste(original_img, (0, 0))
combined_image.paste(depth_colormap, (original_img.width, 0))

# Save the combined image
combined_image.save('combined_image.png')
