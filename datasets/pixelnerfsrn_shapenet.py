import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from torchvision import transforms

from datasets import register


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
        #[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


@register('pixelnerfsrn_shapenet')
class PixelnerfsrnShapenet(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, root_path, split, n_support_views, n_query_views, image_size=(128, 128), world_scale=1.0,
        truncate=None, repeat=1, views_rng=None,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        stage = split
        self.base_path = root_path + "_" + stage
        self.dataset_name = os.path.basename(root_path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False

        self.n_support_views = n_support_views
        self.n_query_views = n_query_views

        if truncate is not None:
            self.intrins = self.intrins[:truncate]
        self.repeat = repeat
        self.views_rng = views_rng

    def __len__(self):
        return len(self.intrins) * self.repeat

    def __getitem__(self, index):
        intrin_path = self.intrins[index % len(self.intrins)]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        if self.views_rng is not None:
            mini, maxi = self.views_rng
            perm = mini + np.random.permutation(maxi - mini)
        else:
            perm = np.random.permutation(len(rgb_paths))
        perm = perm[:self.n_support_views + self.n_query_views]
        rgb_paths = [rgb_paths[_] for _ in perm]
        pose_paths = [pose_paths[_] for _ in perm]

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        c = torch.tensor([cx, cy], dtype=torch.float32)
        # result = {
        #     "path": dir_path,
        #     "img_id": index,
        #     "focal": focal,
        #     "c": c,
        #     "images": all_imgs,
        #     "masks": all_masks,
        #     "bbox": all_bboxes,
        #     "poses": all_poses,
        # }

        num_images = all_poses.shape[0]
        cam_unproj_map = (
            unproj_map(width, height, focal.squeeze(), c=c)
            .unsqueeze(0)
            .repeat(num_images, 1, 1, 1)
        )
        cam_centers = all_poses[:, None, None, :3, 3].expand(-1, height, width, -1)
        cam_raydir = torch.matmul(
            all_poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
        )[:, :, :, :, 0]
        ns = self.n_support_views
        scale = 0.5
        ret = {
            'support_imgs': all_imgs[:ns],
            'support_rays_o': cam_centers[:ns] * scale,
            'support_rays_d': cam_raydir[:ns],
            'query_imgs': all_imgs[ns:],
            'query_rays_o': cam_centers[ns:] * scale,
            'query_rays_d': cam_raydir[ns:],
            'near': torch.tensor(self.z_near, dtype=torch.float32) * scale,
            'far': torch.tensor(self.z_far, dtype=torch.float32) * scale,
        }
        return ret
