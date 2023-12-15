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
img = Image.open(image_path)
img = np.asarray(img, dtype=np.float32) / 255.0

# Preprocess the image
totensor = ToTensor('test')
img = totensor.to_tensor(img)
img = totensor.normalize(img)
img = img.unsqueeze(0).cuda()

# Forward pass through the model
pdepth = model.forward(img)

# Convert depth tensor to a numpy array
pdepth_np = pdepth.squeeze().cpu().detach().numpy()

# Plot the depth map using a colormap
plt.imshow(pdepth_np, cmap='jet')
plt.colorbar()
plt.title('Depth Map')

# Save the plot as an image file (e.g., PNG)
plt.savefig('depth_map.png')