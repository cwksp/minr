import numpy as np
from datasets.celeba import CelebA
from models.siren.sh_base_params import SHBaseParams
from utils.siren import make_coord
from torch.optim import Adam
from torchvision import transforms

img = CelebA('../../data/CelebA', 'train')[0].cuda()
model = SHBaseParams().cuda()

coord = make_coord(img.shape[1:], flatten=True).cuda()
img_flat = img.permute(1, 2, 0).view(-1, 3)

optimizer = Adam(model.parameters(), lr=1e-4)

batch_size = 1024
for i in range(200):
    model.train()
    inds = np.random.choice(len(coord), batch_size, replace=False)
    inds = np.arange(len(coord)) ## disable batch
    coord_, img_flat_ = coord[inds], img_flat[inds]
    pred = model({'support_imgs': img.unsqueeze(0), 'query_coords': coord_.unsqueeze(0)})
    loss = (pred - img_flat_.unsqueeze(0)).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    if (i + 1) % 50 == 0:
        model.eval()
        pred = model({'support_imgs': img.unsqueeze(0), 'query_coords': coord.unsqueeze(0)}).cpu().clip(0, 1)[0]
        transforms.ToPILImage()(pred.view(img.shape[1], img.shape[2], 3).permute(2, 0, 1)).save(f'{i}.png')
