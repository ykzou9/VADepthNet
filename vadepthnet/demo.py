import torch
from PIL import Image
import numpy as np
from vadepthnet.networks.vadepthnet import VADepthNet
from vadepthnet.dataloaders.dataloader import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

model = VADepthNet(max_depth=10,
prior_mean=1.54,
img_size=(480, 640))
model = torch.nn.DataParallel(model)
checkpoint = torch.load('vadepthnet_nyu.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
img = Image.open(image_path)
img = np.asarray(img, dtype=np.float32) / 255.0
#img = torch.from_numpy(img).cuda().unsqueeze(0)

totensor = ToTensor('test')
img = totensor.to_tensor(img)
img = totensor.normalize(img)
img = img.unsqueeze(0)
pdepth = model.forward(img)
print(pdepth)

