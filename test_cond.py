import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import argparse
import os
import time
import numpy as np
from dataset import VirtualTryOnDataset, VirtualTryOnDataLoader
from dataset_test import VirtualTryOnDataset
from VTON_Networks import WarpingFlowGenerator, PerceptualLoss, load_model_checkpoint,  save_model_checkpoint, create_sampling_grid

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from get_norm_const import D_logit


class TestConfigParser:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()

        parser.add_argument("--gpu_ids", default="")
        parser.add_argument('-j', '--workers', type=int, default=4)
        parser.add_argument('-b', '--batch-size', type=int, default=8)
        parser.add_argument('--fp16', action='store_true')

        parser.add_argument("--dataroot", default="./data/zalando-hd-resize")
        parser.add_argument("--datamode", default="test")
        parser.add_argument("--data_list", default="test_pairs.txt")
        parser.add_argument("--datasetting", default="paired")
        parser.add_argument("--fine_width", type=int, default=192)
        parser.add_argument("--fine_height", type=int, default=256)

        parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--tocg_checkpoint', type=str, default='')
        parser.add_argument('--D_checkpoint', type=str, default='')

        parser.add_argument("--tensorboard_count", type=int, default=100)
        parser.add_argument("--shuffle", action='store_true')
        parser.add_argument("--semantic_nc", type=int, default=13)
        parser.add_argument("--output_nc", type=int, default=13)

        parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
        parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")

        parser.add_argument("--clothmask_composition", type=str,
                            choices=['no_composition', 'detach', 'warp_grad'],
                            default='warp_grad')

        parser.add_argument('--upsample', type=str, default='bilinear',
                            choices=['nearest', 'bilinear'])
        parser.add_argument('--occlusion', action='store_true')

        parser.add_argument('--Ddownx2', action='store_true')
        parser.add_argument('--Ddropout', action='store_true')
        parser.add_argument('--num_D', type=int, default=2)
        parser.add_argument('--spectral', action='store_true')
        parser.add_argument('--norm_const', type=float)

        return parser.parse_args()


class OutputManager:
    def __init__(self, config):
        self.config = config
        self.output_base_path = self._create_output_path()

    def _create_output_path(self):
        checkpoint_parts = self.config.tocg_checkpoint.split('/')
        base_path = os.path.join(
            './output',
            checkpoint_parts[-2],
            checkpoint_parts[-1],
            self.config.datamode,
            self.config.datasetting,
            'multi-task'
        )
        os.makedirs(base_path, exist_ok=True)
        return base_path

    def save_visualization(self, grid_image, paired_name, unpaired_name):
        filename = f"{paired_name.split('.')[0]}_{unpaired_name.split('.')[0]}.png"
        save_path = os.path.join(self.output_base_path, filename)
        save_image(grid_image, save_path)

    def save_rejection_scores(self, scores):
        scores.sort(key=lambda x: x[1], reverse=True)
        output_file = os.path.join(self.output_base_path, 'rejection_prob.txt')

        with open(output_file, 'a') as f:
            for name, score in scores:
                f.write(f"{name} {score}\n")


class InferenceEngine:
    def __init__(self, config, generator, discriminator=None):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator

        self.generator.cuda()
        self.generator.eval()

        if self.discriminator is not None:
            self.discriminator.cuda()
            self.discriminator.eval()

    def prepare_inputs(self, batch_data):
        cloth_rgb = batch_data['cloth'][self.config.datasetting].cuda()
        cloth_mask = batch_data['cloth_mask'][self.config.datasetting].cuda()
        cloth_mask = torch.FloatTensor(
            (cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float32)
        ).cuda()

        parse_agnostic = batch_data['parse_agnostic'].cuda()
        densepose = batch_data['densepose'].cuda()

        label_segmap = batch_data['parse'].cuda()
        parse_cloth_mask = batch_data['pcm'].cuda()
        parse_cloth_rgb = batch_data['parse_cloth'].cuda()

        input_cloth = torch.cat([cloth_rgb, cloth_mask], 1)
        input_body = torch.cat([parse_agnostic, densepose], 1)

        return {
            'cloth_rgb': cloth_rgb,
            'cloth_mask': cloth_mask,
            'input_cloth': input_cloth,
            'input_body': input_body,
            'parse_agnostic': parse_agnostic,
            'densepose': densepose,
            'label_segmap': label_segmap,
            'parse_cloth_mask': parse_cloth_mask,
            'parse_cloth_rgb': parse_cloth_rgb,
            'image': batch_data['image'],
            'cloth_names': batch_data['c_name']
        }

    def forward(self, inputs):
        with torch.no_grad():
            flow_list, pred_segmap, warped_cloth, warped_mask = self.generator(
                inputs['input_cloth'],
                inputs['input_body']
            )

            warped_mask_binary = torch.FloatTensor(
                (warped_mask.detach().cpu().numpy() > 0.5).astype(np.float32)
            ).cuda()

            if self.config.clothmask_composition != 'no_composition':
                cloth_channel_mask = torch.ones_like(pred_segmap)

                if self.config.clothmask_composition == 'detach':
                    cloth_channel_mask[:, 3:4, :, :] = warped_mask_binary
                elif self.config.clothmask_composition == 'warp_grad':
                    cloth_channel_mask[:, 3:4, :, :] = warped_mask

                pred_segmap = pred_segmap * cloth_channel_mask

            discriminator_score = None
            if self.discriminator is not None:
                pred_segmap_softmax = F.softmax(pred_segmap, dim=1)
                discriminator_input = torch.cat((
                    inputs['input_cloth'].detach(),
                    inputs['input_body'].detach(),
                    pred_segmap_softmax
                ), dim=1)

                pred_output = self.discriminator(discriminator_input)
                logit_score = D_logit(pred_output)
                discriminator_score = (logit_score / (1 - logit_score)) / self.config.norm_const

            pred_cloth_channel = (torch.argmax(pred_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalignment = pred_cloth_channel - warped_mask_binary
            misalignment[misalignment < 0.0] = 0.0

            return {
                'pred_segmap': pred_segmap,
                'warped_cloth': warped_cloth,
                'warped_mask': warped_mask,
                'warped_mask_binary': warped_mask_binary,
                'misalignment': misalignment,
                'discriminator_score': discriminator_score
            }

    def create_visualization_grid(self, inputs, outputs, batch_idx):
        grid = make_grid([
            (inputs['cloth_rgb'][batch_idx].cpu() / 2 + 0.5),
            (inputs['cloth_mask'][batch_idx].cpu()).expand(3, -1, -1),
            visualize_segmap(inputs['parse_agnostic'].cpu(), batch=batch_idx),
            ((inputs['densepose'].cpu()[batch_idx] + 1) / 2),
            (inputs['parse_cloth_rgb'][batch_idx].cpu() / 2 + 0.5),
            inputs['parse_cloth_mask'][batch_idx].cpu().expand(3, -1, -1),
            (outputs['warped_cloth'][batch_idx].cpu().detach() / 2 + 0.5),
            (outputs['warped_mask_binary'][batch_idx].cpu().detach()).expand(3, -1, -1),
            visualize_segmap(inputs['label_segmap'].cpu(), batch=batch_idx),
            visualize_segmap(outputs['pred_segmap'].cpu(), batch=batch_idx),
            (inputs['image'][batch_idx] / 2 + 0.5),
            (outputs['misalignment'][batch_idx].cpu().detach()).expand(3, -1, -1)
        ], nrow=4)

        return grid


class Tester:
    def __init__(self, config, test_loader, logger, inference_engine, output_manager):
        self.config = config
        self.test_loader = test_loader
        self.logger = logger
        self.inference_engine = inference_engine
        self.output_manager = output_manager

    def run(self):
        start_time = time.time()
        total_samples = 0
        discriminator_scores = []

        for batch_data in self.test_loader.data_loader:
            inputs = self.inference_engine.prepare_inputs(batch_data)
            outputs = self.inference_engine.forward(inputs)

            batch_size = inputs['cloth_rgb'].shape[0]

            if outputs['discriminator_score'] is not None:
                for idx in range(batch_size):
                    cloth_name = inputs['cloth_names']['paired'][idx].replace('.jpg', '.png')
                    score = outputs['discriminator_score'][idx].item()
                    discriminator_scores.append((cloth_name, score))
                    print(f"Rejection probability: {score}")

            for idx in range(batch_size):
                grid = self.inference_engine.create_visualization_grid(inputs, outputs, idx)

                self.output_manager.save_visualization(
                    grid,
                    inputs['cloth_names']['paired'][idx],
                    inputs['cloth_names']['unpaired'][idx]
                )

            total_samples += batch_size
            print(f"Processed: {total_samples} samples")

        if discriminator_scores:
            self.output_manager.save_rejection_scores(discriminator_scores)

        elapsed_time = time.time() - start_time
        print(f"\nTest completed in {elapsed_time:.2f} seconds")
        print(f"Total samples processed: {total_samples}")


def setup_environment(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids

    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)


def create_test_loader(config):
    test_dataset = VirtualTryOnDataset(config)
    test_loader = VirtualTryOnDataLoader(config, test_dataset)
    return test_loader


def create_logger(config):
    checkpoint_parts = config.tocg_checkpoint.split('/')
    log_path = os.path.join(
        config.tensorboard_dir,
        checkpoint_parts[-2],
        checkpoint_parts[-1],
        config.datamode,
        config.datasetting
    )
    logger = SummaryWriter(log_dir=log_path)
    return logger


def create_models(config):
    cloth_input_channels = 4
    body_input_channels = config.semantic_nc + 3

    generator = WarpingFlowGenerator(
        config,
        input1_nc=cloth_input_channels,
        input2_nc=body_input_channels,
        output_nc=config.output_nc,
        ngf=96,
        norm_layer=nn.BatchNorm2d
    )

    discriminator = None
    if config.D_checkpoint and os.path.exists(config.D_checkpoint):
        if config.norm_const is None:
            raise ValueError("norm_const must be provided when using discriminator")

        discriminator = define_D(
            input_nc=cloth_input_channels + body_input_channels + config.output_nc,
            Ddownx2=config.Ddownx2,
            Ddropout=config.Ddropout,
            n_layers_D=3,
            spectral=config.spectral,
            num_D=config.num_D
        )

    if not config.tocg_checkpoint or not os.path.exists(config.tocg_checkpoint):
        raise ValueError(f"Generator checkpoint not found: {config.tocg_checkpoint}")

    print(f"Loading generator from: {config.tocg_checkpoint}")
    load_model_checkpoint(generator, config.tocg_checkpoint)

    if discriminator is not None:
        print(f"Loading discriminator from: {config.D_checkpoint}")
        load_model_checkpoint(discriminator, config.D_checkpoint)

    return generator, discriminator


def main():
    config = TestConfigParser.parse_arguments()

    print("=" * 80)
    print("Testing Configuration")
    print("=" * 80)
    print(config)
    print("=" * 80)

    setup_environment(config)

    print("\n[1/5] Creating test loader...")
    test_loader = create_test_loader(config)
    print(f"✓ Test loader created")

    print("\n[2/5] Initializing logger...")
    logger = create_logger(config)
    print(f"✓ Logger initialized")

    print("\n[3/5] Loading models...")
    generator, discriminator = create_models(config)
    print(f"✓ Generator loaded")
    if discriminator:
        print(f"✓ Discriminator loaded")

    print("\n[4/5] Setting up inference engine...")
    inference_engine = InferenceEngine(config, generator, discriminator)
    output_manager = OutputManager(config)
    print(f"✓ Inference engine ready")
    print(f"✓ Output directory: {output_manager.output_base_path}")

    print("\n[5/5] Starting testing...")
    print(f"Data mode: {config.datamode}")
    print(f"Data setting: {config.datasetting}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 80)

    tester = Tester(config, test_loader, logger, inference_engine, output_manager)
    tester.run()

    logger.close()

    print("\n" + "=" * 80)
    print("✓ Testing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()