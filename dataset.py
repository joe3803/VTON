import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os.path as osp
import numpy as np
import json


class VirtualTryOnDataset(data.Dataset):
    """
    Custom Dataset for Virtual Try-On Model Testing
    """

    def __init__(self, config):
        super(VirtualTryOnDataset, self).__init__()
        self.config = config
        self.root_dir = config.dataroot
        self.mode = config.datamode
        self.data_list_path = config.data_list
        self.img_height = config.fine_height
        self.img_width = config.fine_width
        self.num_semantic_classes = config.semantic_nc
        self.dataset_path = osp.join(config.dataroot, config.datamode)

        # Image normalization transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load image and cloth pairing information
        self.person_imgs, self.garment_imgs = self._load_data_pairs()

    def _load_data_pairs(self):
        """Load person and garment image pairs from data list file"""
        person_list = []
        garment_list = []

        data_file = osp.join(self.root_dir, self.data_list_path)
        with open(data_file, 'r') as f:
            for line in f.readlines():
                person_img, garment_img = line.strip().split()
                person_list.append(person_img)
                garment_list.append(garment_img)

        garment_dict = {
            'paired': person_list,
            'unpaired': garment_list
        }

        return person_list, garment_dict

    def name(self):
        return "VirtualTryOnDataset"

    def _create_body_mask(self, person_img, segmentation_map, keypoints):
        """
        Create body-agnostic representation by masking out clothing regions
        """
        seg_array = np.array(segmentation_map)

        # Extract head region (face + hair)
        head_mask = ((seg_array == 4).astype(np.float32) +
                     (seg_array == 13).astype(np.float32))

        # Extract lower body region (legs + shoes + socks)
        lower_body_mask = ((seg_array == 9).astype(np.float32) +
                           (seg_array == 12).astype(np.float32) +
                           (seg_array == 16).astype(np.float32) +
                           (seg_array == 17).astype(np.float32) +
                           (seg_array == 18).astype(np.float32) +
                           (seg_array == 19).astype(np.float32))

        # Create masked image
        masked_img = person_img.copy()
        draw_tool = ImageDraw.Draw(masked_img)

        # Normalize shoulder width based on hip width
        shoulder_width = np.linalg.norm(keypoints[5] - keypoints[2])
        hip_width = np.linalg.norm(keypoints[12] - keypoints[9])
        hip_center = (keypoints[9] + keypoints[12]) / 2
        keypoints[9] = hip_center + (keypoints[9] - hip_center) / hip_width * shoulder_width
        keypoints[12] = hip_center + (keypoints[12] - hip_center) / hip_width * shoulder_width

        radius = int(shoulder_width / 16) + 1

        # Mask torso region
        for idx in [9, 12]:
            px, py = keypoints[idx]
            draw_tool.ellipse((px - radius * 3, py - radius * 6, px + radius * 3, py + radius * 6), 'gray', 'gray')

        draw_tool.line([tuple(keypoints[i]) for i in [2, 9]], 'gray', width=radius * 6)
        draw_tool.line([tuple(keypoints[i]) for i in [5, 12]], 'gray', width=radius * 6)
        draw_tool.line([tuple(keypoints[i]) for i in [9, 12]], 'gray', width=radius * 12)
        draw_tool.polygon([tuple(keypoints[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # Mask neck region
        nx, ny = keypoints[1]
        draw_tool.rectangle((nx - radius * 5, ny - radius * 9, nx + radius * 5, ny), 'gray', 'gray')

        # Mask shoulders
        draw_tool.line([tuple(keypoints[i]) for i in [2, 5]], 'gray', width=radius * 12)
        for idx in [2, 5]:
            px, py = keypoints[idx]
            draw_tool.ellipse((px - radius * 5, py - radius * 6, px + radius * 5, py + radius * 6), 'gray', 'gray')

        # Mask arms
        for idx in [3, 4, 6, 7]:
            if (keypoints[idx - 1, 0] == 0.0 and keypoints[idx - 1, 1] == 0.0) or \
                    (keypoints[idx, 0] == 0.0 and keypoints[idx, 1] == 0.0):
                continue
            draw_tool.line([tuple(keypoints[j]) for j in [idx - 1, idx]], 'gray', width=radius * 10)
            px, py = keypoints[idx]
            draw_tool.ellipse((px - radius * 5, py - radius * 5, px + radius * 5, py + radius * 5), 'gray', 'gray')

        # Preserve visible arm parts based on segmentation
        for seg_id, kpt_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            arm_mask = Image.new('L', (768, 1024), 'white')
            arm_draw = ImageDraw.Draw(arm_mask)

            px, py = keypoints[kpt_ids[0]]
            arm_draw.ellipse((px - radius * 5, py - radius * 6, px + radius * 5, py + radius * 6), 'black', 'black')

            for idx in kpt_ids[1:]:
                if (keypoints[idx - 1, 0] == 0.0 and keypoints[idx - 1, 1] == 0.0) or \
                        (keypoints[idx, 0] == 0.0 and keypoints[idx, 1] == 0.0):
                    continue
                arm_draw.line([tuple(keypoints[j]) for j in [idx - 1, idx]], 'black', width=radius * 10)
                px, py = keypoints[idx]
                if idx != kpt_ids[-1]:
                    arm_draw.ellipse((px - radius * 5, py - radius * 5, px + radius * 5, py + radius * 5), 'black',
                                     'black')
            arm_draw.ellipse((px - radius * 4, py - radius * 4, px + radius * 4, py + radius * 4), 'black', 'black')

            arm_region = (np.array(arm_mask) / 255) * (seg_array == seg_id).astype(np.float32)
            masked_img.paste(person_img, None, Image.fromarray(np.uint8(arm_region * 255), 'L'))

        # Preserve head and lower body
        masked_img.paste(person_img, None, Image.fromarray(np.uint8(head_mask * 255), 'L'))
        masked_img.paste(person_img, None, Image.fromarray(np.uint8(lower_body_mask * 255), 'L'))

        return masked_img

    def __getitem__(self, idx):
        person_name = self.person_imgs[idx]

        # Load garment images for both paired and unpaired scenarios
        garment_data = {}
        garment_names = {}
        garment_masks = {}

        for pairing_type in self.garment_imgs:
            garment_names[pairing_type] = self.garment_imgs[pairing_type][idx]

            # Load and resize garment
            garment_path = osp.join(self.dataset_path, 'cloth', garment_names[pairing_type])
            garment_img = Image.open(garment_path).convert('RGB')
            garment_img = transforms.Resize(self.img_width, interpolation=2)(garment_img)
            garment_data[pairing_type] = self.normalize(garment_img)

            # Load and process garment mask
            mask_path = osp.join(self.dataset_path, 'cloth-mask', garment_names[pairing_type])
            mask_img = Image.open(mask_path)
            mask_img = transforms.Resize(self.img_width, interpolation=0)(mask_img)
            mask_array = np.array(mask_img)
            mask_binary = (mask_array >= 128).astype(np.float32)
            garment_masks[pairing_type] = torch.from_numpy(mask_binary).unsqueeze(0)

        # Load person image
        person_path = osp.join(self.dataset_path, 'image', person_name)
        person_img_orig = Image.open(person_path)
        person_img = transforms.Resize(self.img_width, interpolation=2)(person_img_orig)
        person_tensor = self.normalize(person_img)

        # Load segmentation map
        seg_name = person_name.replace('.jpg', '.png')
        seg_path = osp.join(self.dataset_path, 'image-parse-v3', seg_name)
        seg_img_orig = Image.open(seg_path)
        seg_img = transforms.Resize(self.img_width, interpolation=0)(seg_img_orig)
        seg_tensor = torch.from_numpy(np.array(seg_img)[None]).long()
        seg_normalized = self.normalize(seg_img.convert('RGB'))

        # Define body part label mappings
        body_part_labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper_cloth', [5, 6, 7]],
            4: ['pants', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        # Create one-hot encoded segmentation maps
        seg_onehot_20 = torch.FloatTensor(20, self.img_height, self.img_width).zero_()
        seg_onehot_20 = seg_onehot_20.scatter_(0, seg_tensor, 1.0)

        seg_onehot_compressed = torch.FloatTensor(self.num_semantic_classes, self.img_height, self.img_width).zero_()
        for i in range(len(body_part_labels)):
            for label_id in body_part_labels[i][1]:
                seg_onehot_compressed[i] += seg_onehot_20[label_id]

        # Create single-channel segmentation for visualization
        seg_single_channel = torch.FloatTensor(1, self.img_height, self.img_width).zero_()
        for i in range(len(body_part_labels)):
            for label_id in body_part_labels[i][1]:
                seg_single_channel[0] += seg_onehot_20[label_id] * i

        # Load body-agnostic segmentation
        agnostic_seg_path = osp.join(self.dataset_path, 'image-parse-agnostic-v3.2', seg_name)
        agnostic_seg_img = Image.open(agnostic_seg_path)
        agnostic_seg_img = transforms.Resize(self.img_width, interpolation=0)(agnostic_seg_img)
        agnostic_seg_tensor = torch.from_numpy(np.array(agnostic_seg_img)[None]).long()
        agnostic_seg_normalized = self.normalize(agnostic_seg_img.convert('RGB'))

        agnostic_seg_onehot = torch.FloatTensor(20, self.img_height, self.img_width).zero_()
        agnostic_seg_onehot = agnostic_seg_onehot.scatter_(0, agnostic_seg_tensor, 1.0)

        agnostic_seg_compressed = torch.FloatTensor(self.num_semantic_classes, self.img_height, self.img_width).zero_()
        for i in range(len(body_part_labels)):
            for label_id in body_part_labels[i][1]:
                agnostic_seg_compressed[i] += agnostic_seg_onehot[label_id]

        # Extract cloth region from person image
        cloth_region_mask = seg_onehot_compressed[3:4]
        person_cloth_region = person_tensor * cloth_region_mask + (1 - cloth_region_mask)

        # Load pose keypoints visualization
        pose_vis_name = person_name.replace('.jpg', '_rendered.png')
        pose_vis_path = osp.join(self.dataset_path, 'openpose_img', pose_vis_name)
        pose_vis = Image.open(pose_vis_path)
        pose_vis = transforms.Resize(self.img_width, interpolation=2)(pose_vis)
        pose_vis_tensor = self.normalize(pose_vis)

        # Load pose keypoints data
        pose_json_name = person_name.replace('.jpg', '_keypoints.json')
        pose_json_path = osp.join(self.dataset_path, 'openpose_json', pose_json_name)
        with open(pose_json_path, 'r') as f:
            pose_json = json.load(f)
            keypoints_flat = pose_json['people'][0]['pose_keypoints_2d']
            keypoints = np.array(keypoints_flat).reshape((-1, 3))[:, :2]

        # Load DensePose
        densepose_name = person_name.replace('image', 'image-densepose')
        densepose_path = osp.join(self.dataset_path, 'image-densepose', densepose_name)
        densepose_img = Image.open(densepose_path)
        densepose_img = transforms.Resize(self.img_width, interpolation=2)(densepose_img)
        densepose_tensor = self.normalize(densepose_img)

        # Create body-agnostic representation
        body_agnostic = self._create_body_mask(person_img_orig, seg_img_orig, keypoints)
        body_agnostic = transforms.Resize(self.img_width, interpolation=2)(body_agnostic)
        body_agnostic_tensor = self.normalize(body_agnostic)

        return {
            # Metadata
            'garment_name': garment_names,
            'person_name': person_name,

            # Input: Garment
            'garment': garment_data,
            'garment_mask': garment_masks,

            # Input: Body representation
            'body_seg_agnostic': agnostic_seg_compressed,
            'densepose': densepose_tensor,
            'pose_keypoints_vis': pose_vis_tensor,

            # Ground truth
            'seg_onehot': seg_single_channel,
            'seg_map': seg_onehot_compressed,
            'cloth_region_mask': cloth_region_mask,
            'person_cloth_region': person_cloth_region,

            # Visualization
            'person_img': person_tensor,
            'body_agnostic': body_agnostic_tensor
        }

    def __len__(self):
        return len(self.person_imgs)


class VirtualTryOnDataLoader:
    """
    Custom DataLoader wrapper for Virtual Try-On Dataset
    """

    def __init__(self, config, dataset):
        super(VirtualTryOnDataLoader, self).__init__()

        sampler = None
        shuffle = False

        if config.shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            shuffle = False
        else:
            shuffle = False

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )

        self.dataset = dataset
        self.iterator = iter(self.loader)

    def get_next_batch(self):
        """Get next batch, automatically restarting iterator when exhausted"""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch