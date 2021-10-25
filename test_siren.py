import numpy as np
from datasets.celeba import CelebA
from models.siren.siren import SirenModel
from utils.siren import make_coord
from torch.optim import Adam
from torchvision import transforms

img = CelebA('../../data/CelebA', 'train')[0]
model = SirenModel(w0=200).cuda()

coord = make_coord(img.shape[1:], flatten=True).cuda()
img_flat = img.permute(1, 2, 0).view(-1, 3).cuda()

optimizer = Adam(model.parameters(), lr=1e-4)

batch_size = 1024
for i in range(200):
    model.train()
    inds = np.random.choice(len(coord), batch_size, replace=False)
    inds = np.arange(len(coord)) ## disable batch
    coord_, img_flat_ = coord[inds], img_flat[inds]
    pred = model(coord_)
    loss = (pred - img_flat_).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    if (i + 1) % 20 == 0:
        model.eval()
        pred = model(coord).cpu().clip(0, 1)
        transforms.ToPILImage()(pred.view(img.shape[1], img.shape[2], 3).permute(2, 0, 1)).save(f'{i}.png')
