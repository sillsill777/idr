import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,  # False
                 data_dir,  # DTU
                 img_res, # [1200,1600]
                 scan_id=0,  # 65
                 cam_file=None
                 ):

        self.instance_dir = os.path.join('../../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        # scale_mat, scale_mat_inv, world_mat, world_mat_inv, camera_mat, camera_mat_inv
        '''
        f=open(self.instance_dir+'/camera.txt','w')
        i=0
        for key,val in camera_dict.items():
            i+=1
            f.write(str(key) + ': \n')
            f.write(str(val))
            f.write('\n\n')
            if not i%6 and i != 0:
                f.write('='*70+'\n')
        f.close()
        return
        '''
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]  # (4,4)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]  # (4,4)

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)  # (3,1200,1600) 3, H , W
            rgb = rgb.reshape(3, -1).transpose(1, 0)  # (1920000, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        self.object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            object_mask = object_mask.reshape(-1)  # (1920000,)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        '''
        img_res=[5,3]
        uv=[[0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [0, 3],
            [1, 3],
            [2, 3],
            [0, 4],
            [1, 4],
            [2, 4]]
        '''

        sample = {
            "object_mask": self.object_masks[idx],  # torch.Size([1920000])
            "uv": uv,  # torch.Size([1920000, 2])
            "intrinsics": self.intrinsics_all[idx],  # torch.Size([4, 4])
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]  # torch.Size([1920000, 3])
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]  # torch.Size([4, 4])

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        #  [print(i,'\n\n') for i in batch_list[0]]
        """'
        batch_list는 batch_size 만큼의 튜플을 원소 개수로 가지는 리스트
        각 튜플은
        (idx,  {'object_mask':tensor(...), 'uv':tensor(...),
        'intrinsics':tensor(...), 'pose':tensor(...)},  {'rgb':tensor(...)})임
        """
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)
        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
        '''
        for val in all_parsed:
            print(val,'\n')
            if type(val)==dict:
                for key,v in val.items():
                    print(key+': \n')
                    print(v,'\n')
                    print(v.shape)
        '''
        '''
        return ( idx->[9,11],  sample->{'object_mask':tensor(batch, ...), 'uv':tensor(batch, ...),
        'intrinsics':tensor(batch, ...), 'pose':tensor(batch, ...)},  ground_truth->{'rgb':tensor(batch,...)} )
        '''
        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat


if __name__ == '__main__':
    ds = SceneDataset(False, 'DTU', [1200, 1600], 65)
    idx1, sample1, ground_truth1=ds.__getitem__(23)
    for key,val in sample1.items():
        print(key+' : \n')
        print(str(val)+'\n')
        print('shape: ')
        print(val.shape)
    print(ground_truth1)
    print(ground_truth1['rgb'].shape)
    print('='*70)
    a = torch.utils.data.DataLoader(ds,
                                    batch_size=2,
                                    shuffle=True,
                                    collate_fn=ds.collate_fn)
    x=next(iter(a))
    indices, model_input, ground_truth=x
    print(indices.shape)
    for key,val in model_input.items():
        print(key)
        print(val.shape)
    for key,val in ground_truth.items():
        print(key)
        print(val.shape)