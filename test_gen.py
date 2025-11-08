import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import torchgeometry as tgm

import argparse
import os
import time
import numpy as np
from collections import OrderedDict

from dataset import VirtualTryOnDataset, VirtualTryOnDataLoader
from dataset_test import VirtualTryOnDataset
from VTON_Networks import WarpingFlowGenerator, PerceptualLoss, load_model_checkpoint,  save_model_checkpoint, create_sampling_grid
from utils import save_images, visualize_segmap


class InferenceConfig:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--gpu_ids", default="")
        parser.add_argument('-j', '--workers', type=int, default=4)
        parser.add_argument('-b', '--batch-size', type=int, default=1)
        parser.add_argument('--fp16', action='store_true', help='use amp')
        parser.add_argument('--cuda', default=False, help='cuda or cpu')

        parser.add_argument('--test_name', type=str, default='test')
        parser.add_argument("--dataroot", default="./data/zalando-hd-resize")
        parser.add_argument("--datamode", default="test")
        parser.add_argument("--data_list", default="test_pairs.txt")
        parser.add_argument("--output_dir", type=str, default="./Output")
        parser.add_argument("--datasetting", default="unpaired")
        parser.add_argument("--fine_width", type=int, default=768)
        parser.add_argument("--fine_height", type=int, default=1024)

        parser.add_argument('--tensorboard_dir', type=str, default='./data/zalando-hd-resize/tensorboard')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--tocg_checkpoint', type=str, default='./eval_models/weights/v0.1/mtviton.pth')
        parser.add_argument('--gen_checkpoint', type=str, default='./eval_models/weights/v0.1/gen.pth')

        parser.add_argument("--tensorboard_count", type=int, default=100)
        parser.add_argument("--shuffle", action='store_true')
        parser.add_argument("--semantic_nc", type=int, default=13)
        parser.add_argument("--output_nc", type=int, default=13)
        parser.add_argument('--gen_semantic_nc', type=int, default=7)

        parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
        parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
        parser.add_argument("--clothmask_composition", type=str,
                            choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')

        parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
        parser.add_argument('--occlusion', action='store_true')

        parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--init_type', type=str, default='xavier')
        parser.add_argument('--init_variance', type=float, default=0.02)
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='most')

        self.config = parser.parse_args()

    def get(self):
        return self.config


class ClothWarpingProcessor:
    def __init__(self, config):
        self.config = config
        self.gaussian_blur = tgm.image.GaussianBlur((15, 15), (3, 3))
        if config.cuda:
            self.gaussian_blur = self.gaussian_blur.cuda()

    def remove_overlap(self, segmentation, warped_mask):
        assert len(warped_mask.shape) == 4
        overlap_region = torch.cat([
            segmentation[:, 1:3, :, :],
            segmentation[:, 5:, :, :]
        ], dim=1)
        overlap_sum = overlap_region.sum(dim=1, keepdim=True)
        return warped_mask - overlap_sum * warped_mask

    def process_warping(self, batch_inputs, warping_network):
        device = 'cuda' if self.config.cuda else 'cpu'

        densepose = batch_inputs['densepose'].to(device) if self.config.cuda else batch_inputs['densepose']
        cloth_mask = batch_inputs['cloth_mask'][self.config.datasetting]
        agnostic_parse = batch_inputs['parse_agnostic']
        person_representation = batch_inputs['agnostic'].to(device) if self.config.cuda else batch_inputs['agnostic']
        cloth_image = batch_inputs['cloth'][self.config.datasetting].to(device) if self.config.cuda else \
        batch_inputs['cloth'][self.config.datasetting]
        target_image = batch_inputs['image']
        pose_map = batch_inputs['pose'].to(device) if self.config.cuda else batch_inputs['pose']

        if self.config.cuda:
            cloth_mask = torch.FloatTensor(
                (cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float)
            ).cuda()
            agnostic_parse = agnostic_parse.cuda()
        else:
            cloth_mask = torch.FloatTensor(
                (cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float)
            )

        downsampled_mask = F.interpolate(cloth_mask, size=(256, 192), mode='nearest')
        downsampled_parse = F.interpolate(agnostic_parse, size=(256, 192), mode='nearest')
        downsampled_cloth = F.interpolate(cloth_image, size=(256, 192), mode='bilinear')
        downsampled_pose = F.interpolate(densepose, size=(256, 192), mode='bilinear')

        cloth_input = torch.cat([downsampled_cloth, downsampled_mask], 1)
        context_input = torch.cat([downsampled_parse, downsampled_pose], 1)

        flow_list, segmentation_pred, warped_cloth_low, warped_mask_low = warping_network(
            self.config, cloth_input, context_input
        )

        if self.config.cuda:
            mask_binary = torch.FloatTensor(
                (warped_mask_low.detach().cpu().numpy() > 0.5).astype(np.float)
            ).cuda()
        else:
            mask_binary = torch.FloatTensor(
                (warped_mask_low.detach().cpu().numpy() > 0.5).astype(np.float)
            )

        if self.config.clothmask_composition != 'no_composition':
            mask_modifier = torch.ones_like(segmentation_pred)

            if self.config.clothmask_composition == 'detach':
                mask_modifier[:, 3:4, :, :] = mask_binary
            elif self.config.clothmask_composition == 'warp_grad':
                mask_modifier[:, 3:4, :, :] = warped_mask_low

            segmentation_pred = segmentation_pred * mask_modifier

        blurred_segmentation = self.gaussian_blur(
            F.interpolate(segmentation_pred, size=(self.config.fine_height, self.config.fine_width), mode='bilinear')
        )
        parse_indices = blurred_segmentation.argmax(dim=1)[:, None]

        if self.config.cuda:
            old_parse = torch.FloatTensor(
                parse_indices.size(0), 13, self.config.fine_height, self.config.fine_width
            ).zero_().cuda()
        else:
            old_parse = torch.FloatTensor(
                parse_indices.size(0), 13, self.config.fine_height, self.config.fine_width
            ).zero_()
        old_parse.scatter_(1, parse_indices, 1.0)

        label_mapping = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }

        if self.config.cuda:
            parse_map = torch.FloatTensor(
                parse_indices.size(0), 7, self.config.fine_height, self.config.fine_width
            ).zero_().cuda()
        else:
            parse_map = torch.FloatTensor(
                parse_indices.size(0), 7, self.config.fine_height, self.config.fine_width
            ).zero_()

        for idx in range(len(label_mapping)):
            for label in label_mapping[idx][1]:
                parse_map[:, idx] += old_parse[:, label]

        batch_size, _, height, width = cloth_image.shape
        flow = F.interpolate(
            flow_list[-1].permute(0, 3, 1, 2),
            size=(height, width),
            mode='bilinear'
        ).permute(0, 2, 3, 1)

        flow_normalized = torch.cat([
            flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0),
            flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)
        ], 3)

        sampling_grid = make_grid(batch_size, height, width, self.config)
        warped_grid = sampling_grid + flow_normalized
        warped_cloth = F.grid_sample(cloth_image, warped_grid, padding_mode='border')
        warped_mask = F.grid_sample(cloth_mask, warped_grid, padding_mode='border')

        if self.config.occlusion:
            warped_mask = self.remove_overlap(F.softmax(blurred_segmentation, dim=1), warped_mask)
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)

        return {
            'person_rep': person_representation,
            'densepose': densepose,
            'warped_cloth': warped_cloth,
            'warped_mask': warped_mask,
            'parse_map': parse_map,
            'blurred_seg': blurred_segmentation,
            'agnostic_parse': agnostic_parse,
            'cloth': cloth_image,
            'cloth_mask': cloth_mask,
            'pose_map': pose_map,
            'target': target_image,
            'batch_size': batch_size
        }


class ModelInference:
    def __init__(self, config, warping_network, generator):
        self.config = config
        self.warping_network = warping_network
        self.generator = generator
        self.cloth_processor = ClothWarpingProcessor(config)

        if config.cuda:
            self.warping_network.cuda()
        self.warping_network.eval()
        self.generator.eval()

    def setup_output_directories(self):
        if self.config.output_dir is not None:
            output_dir = self.config.output_dir
        else:
            output_dir = os.path.join(
                './output', self.config.test_name,
                self.config.datamode, self.config.datasetting,
                'generator', 'output'
            )

        grid_dir = os.path.join(
            './output', self.config.test_name,
            self.config.datamode, self.config.datasetting,
            'generator', 'grid'
        )

        os.makedirs(grid_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        return output_dir, grid_dir

    def create_visualization_grid(self, processed_data, generated_output, batch_index):
        grid = make_image_grid([
            (processed_data['cloth'][batch_index].cpu() / 2 + 0.5),
            (processed_data['cloth_mask'][batch_index].cpu()).expand(3, -1, -1),
            visualize_segmap(processed_data['agnostic_parse'].cpu(), batch=batch_index),
            ((processed_data['densepose'].cpu()[batch_index] + 1) / 2),
            (processed_data['warped_cloth'][batch_index].cpu().detach() / 2 + 0.5),
            (processed_data['warped_mask'][batch_index].cpu().detach()).expand(3, -1, -1),
            visualize_segmap(processed_data['blurred_seg'].cpu(), batch=batch_index),
            (processed_data['pose_map'][batch_index].cpu() / 2 + 0.5),
            (processed_data['warped_cloth'][batch_index].cpu() / 2 + 0.5),
            (processed_data['person_rep'][batch_index].cpu() / 2 + 0.5),
            (processed_data['target'][batch_index] / 2 + 0.5),
            (generated_output[batch_index].cpu() / 2 + 0.5)
        ], nrow=4)

        return grid

    def run_inference(self, data_loader):
        output_dir, grid_dir = self.setup_output_directories()

        total_processed = 0
        start_time = time.time()

        with torch.no_grad():
            for batch_inputs in data_loader.data_loader:
                processed_data = self.cloth_processor.process_warping(
                    batch_inputs, self.warping_network
                )

                generator_input = torch.cat((
                    processed_data['person_rep'],
                    processed_data['densepose'],
                    processed_data['warped_cloth']
                ), dim=1)

                generated_output = self.generator(generator_input, processed_data['parse_map'])

                output_names = []
                for i in range(processed_data['batch_size']):
                    grid = self.create_visualization_grid(processed_data, generated_output, i)

                    paired_name = batch_inputs['c_name']['paired'][i].split('.')[0]
                    current_name = batch_inputs['c_name'][self.config.datasetting][i].split('.')[0]
                    output_filename = f'{paired_name}_{current_name}.png'

                    save_image(grid, os.path.join(grid_dir, output_filename))
                    output_names.append(output_filename)

                save_images(generated_output, output_names, output_dir)

                total_processed += processed_data['batch_size']
                print(total_processed)

        elapsed_time = time.time() - start_time
        print(f"Test time: {elapsed_time:.2f}s")


def load_generator_checkpoint(model, checkpoint_path, config):
    if not os.path.exists(checkpoint_path):
        print("Invalid checkpoint path!")
        return

    state_dict = torch.load(checkpoint_path)

    new_state_dict = OrderedDict([
        (k.replace('ace', 'alias').replace('.Spade', ''), v)
        for (k, v) in state_dict.items()
    ])
    new_state_dict._metadata = OrderedDict([
        (k.replace('ace', 'alias').replace('.Spade', ''), v)
        for (k, v) in state_dict._metadata.items()
    ])

    model.load_state_dict(new_state_dict, strict=True)

    if config.cuda:
        model.cuda()


def main():
    config_manager = InferenceConfig()
    config = config_manager.get()
    print(config)
    print("Starting inference...")

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids

    test_dataset = CPDatasetTest(config)
    test_loader = CPDataLoader(config, test_dataset)

    cloth_channels = 4
    context_channels = config.semantic_nc + 3
    output_channels = config.output_nc
    feature_channels = 96

    warping_network = ConditionGenerator(
        config,
        input1_nc=cloth_channels,
        input2_nc=context_channels,
        output_nc=output_channels,
        ngf=feature_channels,
        norm_layer=nn.BatchNorm2d
    )

    config.semantic_nc = 7
    generator = SPADEGenerator(config, 3 + 3 + 3)
    generator.print_network()

    load_checkpoint(warping_network, config.tocg_checkpoint, config)
    load_generator_checkpoint(generator, config.gen_checkpoint, config)

    inference_engine = ModelInference(config, warping_network, generator)
    inference_engine.run_inference(test_loader)

    print("Inference completed!")


if __name__ == "__main__":
    main()