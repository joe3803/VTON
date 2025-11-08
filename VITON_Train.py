import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import Subset

# Import refactored modules (update these import names based on your refactored files)
from dataset import VirtualTryOnDataset, VirtualTryOnDataset, VirtualTryOnDataLoader
from VTON_Networks import (
    WarpingFlowGenerator, PerceptualLoss, AdversarialLossFunction,
    load_model_checkpoint, save_model_checkpoint, create_discriminator,
    create_sampling_grid
)
from utils import visualize_segmap, cross_entropy2d


class TrainingConfig:
    """Configuration for training virtual try-on model"""

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Train Virtual Try-On Model')

        # Experiment settings
        parser.add_argument("--name", default="experiment", help="Experiment name")
        parser.add_argument("--gpu_ids", default="", help="GPU IDs to use")
        parser.add_argument('-j', '--workers', type=int, default=4, help="Number of data loading workers")
        parser.add_argument('-b', '--batch-size', type=int, default=8, help="Batch size")
        parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')

        # Dataset settings
        parser.add_argument("--dataroot", default="./data/", help="Dataset root directory")
        parser.add_argument("--datamode", default="train", help="Dataset mode (train/test)")
        parser.add_argument("--data_list", default="train_pairs.txt", help="Data pairs file")
        parser.add_argument("--fine_width", type=int, default=192, help="Image width")
        parser.add_argument("--fine_height", type=int, default=256, help="Image height")

        # Logging and checkpointing
        parser.add_argument('--tensorboard_dir', type=str, default='tensorboard',
                            help='TensorBoard log directory')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                            help='Checkpoint save directory')
        parser.add_argument('--checkpoint_path', type=str, default='',
                            help='Path to load checkpoint')

        # Training schedule
        parser.add_argument("--tensorboard_count", type=int, default=100,
                            help="Log to TensorBoard every N steps")
        parser.add_argument("--display_count", type=int, default=100,
                            help="Print to console every N steps")
        parser.add_argument("--save_count", type=int, default=10000,
                            help="Save checkpoint every N steps")
        parser.add_argument("--load_step", type=int, default=0,
                            help="Starting step number")
        parser.add_argument("--keep_step", type=int, default=300000,
                            help="Total training steps")
        parser.add_argument("--val_count", type=int, default=1000,
                            help="Validate every N steps")

        # Data settings
        parser.add_argument("--shuffle", action='store_true', help='Shuffle training data')
        parser.add_argument("--semantic_nc", type=int, default=13,
                            help="Number of semantic classes")
        parser.add_argument("--output_nc", type=int, default=13,
                            help="Number of output channels")

        # Network architecture
        parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1",
                            help="Feature warping strategy")
        parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu",
                            help="Output layer type")
        parser.add_argument('--num_D', type=int, default=2,
                            help='Number of discriminator scales')
        parser.add_argument('--Ddownx2', action='store_true',
                            help="Downsample discriminator input")
        parser.add_argument('--Ddropout', action='store_true',
                            help="Apply dropout to discriminator")
        parser.add_argument('--spectral', action='store_true',
                            help="Apply spectral normalization to discriminator")

        # Hardware
        parser.add_argument('--cuda', default=False, help='Use CUDA or CPU')

        # Training strategy
        parser.add_argument("--train_separately", action='store_true',
                            help="Train generator and discriminator separately")
        parser.add_argument("--no_GAN_loss", action='store_true',
                            help="Disable GAN loss")
        parser.add_argument("--last_tv_only", action='store_true',
                            help="Apply TV loss only to final flow")
        parser.add_argument("--intermediate_flow_loss", action='store_true',
                            help="Apply loss to intermediate flow predictions")
        parser.add_argument("--clothmask_composition", type=str,
                            choices=['no_composition', 'detach', 'warp_grad'],
                            default='warp_grad',
                            help="Cloth mask composition strategy")
        parser.add_argument('--edge_aware_tv', type=str,
                            choices=['no_edge', 'last_only', 'weighted'],
                            default="no_edge",
                            help="Edge-aware TV loss strategy")
        parser.add_argument('--add_last_tv', action='store_true',
                            help="Add additional TV loss for final flow")
        parser.add_argument('--occlusion', action='store_true',
                            help="Handle occlusions")

        # Test visualization
        parser.add_argument("--no_test_visualize", action='store_true',
                            help="Disable test visualization")
        parser.add_argument("--num_test_visualize", type=int, default=3,
                            help="Number of test samples to visualize")
        parser.add_argument("--test_datasetting", default="unpaired",
                            help="Test data pairing setting")
        parser.add_argument("--test_dataroot", default="./data/",
                            help="Test dataset root")
        parser.add_argument("--test_data_list", default="test_pairs.txt",
                            help="Test data pairs file")

        # Hyperparameters
        parser.add_argument('--G_lr', type=float, default=0.0002,
                            help='Generator learning rate')
        parser.add_argument('--D_lr', type=float, default=0.0002,
                            help='Discriminator learning rate')
        parser.add_argument('--CE_lambda', type=float, default=10,
                            help='Cross-entropy loss weight')
        parser.add_argument('--GAN_lambda', type=float, default=1,
                            help='GAN loss weight')
        parser.add_argument('--tv_lambda', type=float, default=2,
                            help='Total variation loss weight')
        parser.add_argument('--upsample', type=str, default='bilinear',
                            choices=['nearest', 'bilinear'],
                            help='Upsampling mode')

        config = parser.parse_args()
        return config


def compute_iou_metric(predictions, targets):
    """
    Compute Intersection over Union (IoU) metric

    Args:
        predictions: Predicted segmentation maps (batch_size, H, W)
        targets: Target segmentation maps (batch_size, H, W)

    Returns:
        Mean IoU across batch
    """
    batch_size = predictions.shape[0]
    total_iou = 0

    for i in range(batch_size):
        pred = predictions[i]
        target = targets[i]

        # Threshold predictions
        pred_binary = pred > 0.5
        pred_flat = pred_binary.flatten()
        target_flat = target.flatten()

        # Compute IoU
        intersection = torch.sum(pred_flat[target_flat == 1])
        union = torch.sum(pred_flat) + torch.sum(target_flat)

        iou = (intersection + 1e-7) / (union - intersection + 1e-7)
        total_iou += iou / batch_size

    return total_iou


def remove_occlusion_overlap(segmentation_output, warped_cloth_mask):
    """
    Remove overlapping regions between warped cloth and other body parts

    Args:
        segmentation_output: Predicted segmentation map
        warped_cloth_mask: Warped cloth mask

    Returns:
        Adjusted cloth mask without overlaps
    """
    assert len(warped_cloth_mask.shape) == 4, "Cloth mask must be 4D tensor"

    # Remove overlap with other body parts (excluding background and cloth)
    other_parts = torch.cat([
        segmentation_output[:, 1:3, :, :],  # Hair and face
        segmentation_output[:, 5:, :, :]  # Arms, legs, etc.
    ], dim=1)

    overlap_mask = other_parts.sum(dim=1, keepdim=True) * warped_cloth_mask
    adjusted_mask = warped_cloth_mask - overlap_mask

    return adjusted_mask


def compute_total_variation_loss(flow_list, config, warped_cloth_mask=None):
    """
    Compute total variation loss for optical flow smoothness

    Args:
        flow_list: List of flow predictions at different scales
        config: Training configuration
        warped_cloth_mask: Optional mask for edge-aware TV loss

    Returns:
        Total variation loss
    """
    total_tv_loss = 0

    if config.edge_aware_tv == 'no_edge':
        # Standard TV loss without edge awareness
        flow_range = flow_list if not config.last_tv_only else flow_list[-1:]

        for flow in flow_range:
            tv_y = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
            tv_x = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
            total_tv_loss += tv_y + tv_x

    else:
        # Edge-aware TV loss
        if config.edge_aware_tv == 'last_only':
            flow = flow_list[-1]
            downsampled_mask = F.interpolate(
                warped_cloth_mask,
                size=flow.shape[1:3],
                mode='bilinear'
            )

            # Compute TV with edge weights
            tv_y = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
            tv_x = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

            mask_permuted = downsampled_mask.permute(0, 2, 3, 1)
            weight_y = torch.exp(-150 * torch.abs(
                mask_permuted[:, 1:, :, :] - mask_permuted[:, :-1, :, :]
            ))
            weight_x = torch.exp(-150 * torch.abs(
                mask_permuted[:, :, 1:, :] - mask_permuted[:, :, :-1, :]
            ))

            tv_y = (tv_y * weight_y).mean()
            tv_x = (tv_x * weight_x).mean()
            total_tv_loss += tv_y + tv_x

        elif config.edge_aware_tv == 'weighted':
            # Weighted TV loss across pyramid levels
            for level_idx in range(5):
                flow = flow_list[level_idx]
                downsampled_mask = F.interpolate(
                    warped_cloth_mask,
                    size=flow.shape[1:3],
                    mode='bilinear'
                )

                tv_y = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                tv_x = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

                mask_permuted = downsampled_mask.permute(0, 2, 3, 1)
                weight_y = torch.exp(-150 * torch.abs(
                    mask_permuted[:, 1:, :, :] - mask_permuted[:, :-1, :, :]
                ))
                weight_x = torch.exp(-150 * torch.abs(
                    mask_permuted[:, :, 1:, :] - mask_permuted[:, :, :-1, :]
                ))

                # Scale by pyramid level
                scale_factor = 2 ** (4 - level_idx)
                tv_y = (tv_y * weight_y).mean() / scale_factor
                tv_x = (tv_x * weight_x).mean() / scale_factor
                total_tv_loss += tv_y + tv_x

        # Additional standard TV loss if configured
        if config.add_last_tv:
            flow = flow_list[-1]
            tv_y = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
            tv_x = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
            total_tv_loss += tv_y + tv_x

    return total_tv_loss


def compute_intermediate_flow_losses(flow_list, cloth_input, cloth_mask,
                                     target_cloth, segmentation_output, config,
                                     l1_criterion, vgg_criterion):
    """
    Compute losses for intermediate flow predictions

    Args:
        flow_list: List of flow predictions
        cloth_input: Input cloth image
        cloth_mask: Input cloth mask
        target_cloth: Target cloth region
        segmentation_output: Predicted segmentation
        config: Training configuration
        l1_criterion: L1 loss function
        vgg_criterion: VGG perceptual loss function

    Returns:
        Accumulated L1 and VGG losses
    """
    batch_size, _, height, width = cloth_input.size()
    total_l1_loss = 0
    total_vgg_loss = 0

    # Process all flows except the last one
    for level_idx in range(len(flow_list) - 1):
        flow = flow_list[level_idx]
        _, flow_h, flow_w, _ = flow.size()

        # Create sampling grid
        sampling_grid = create_sampling_grid(batch_size, height, width, config)

        # Upsample flow to input resolution
        upsampled_flow = F.interpolate(
            flow.permute(0, 3, 1, 2),
            size=cloth_input.shape[2:],
            mode=config.upsample
        ).permute(0, 2, 3, 1)

        # Normalize flow
        flow_normalized = torch.cat([
            upsampled_flow[:, :, :, 0:1] / ((flow_w - 1.0) / 2.0),
            upsampled_flow[:, :, :, 1:2] / ((flow_h - 1.0) / 2.0)
        ], dim=3)

        # Warp cloth and mask
        warped_cloth = F.grid_sample(
            cloth_input,
            flow_normalized + sampling_grid,
            padding_mode='border'
        )
        warped_mask = F.grid_sample(
            cloth_mask,
            flow_normalized + sampling_grid,
            padding_mode='border'
        )

        # Remove occlusions
        warped_mask = remove_occlusion_overlap(
            F.softmax(segmentation_output, dim=1),
            warped_mask
        )

        # Compute losses with pyramid weighting
        scale_weight = 2 ** (4 - level_idx)
        total_l1_loss += l1_criterion(warped_mask, target_cloth) / scale_weight
        total_vgg_loss += vgg_criterion(warped_cloth, target_cloth) / scale_weight

    return total_l1_loss, total_vgg_loss


class VirtualTryOnTrainer:
    """Main trainer for virtual try-on model"""

    def __init__(self, config):
        self.config = config
        self.setup_models()
        self.setup_criteria()
        self.setup_optimizers()

    def setup_models(self):
        """Initialize generator and discriminator"""
        # Generator configuration
        cloth_channels = 4  # RGB + mask
        pose_channels = self.config.semantic_nc + 3  # Segmentation + densepose

        self.generator = WarpingFlowGenerator(
            config=self.config,
            cloth_channels=cloth_channels,
            pose_channels=pose_channels,
            output_channels=self.config.output_nc,
            base_filters=96,
            norm_layer=nn.BatchNorm2d
        )

        # Discriminator configuration
        self.discriminator = create_discriminator(
            input_channels=cloth_channels + pose_channels + self.config.output_nc,
            downsample_input=self.config.Ddownx2,
            use_dropout=self.config.Ddropout,
            num_layers=3,
            use_spectral_norm=self.config.spectral,
            num_scales=self.config.num_D
        )

        # Move to GPU
        self.generator.cuda()
        self.generator.train()
        self.discriminator.cuda()
        self.discriminator.train()

    def setup_criteria(self):
        """Initialize loss functions"""
        self.l1_criterion = nn.L1Loss()
        self.vgg_criterion = PerceptualLoss(self.config)

        if self.config.fp16:
            self.gan_criterion = AdversarialLossFunction(
                use_least_squares=True,
                tensor_type=torch.cuda.HalfTensor
            )
        else:
            tensor_type = torch.cuda.FloatTensor if self.config.cuda else torch.FloatTensor
            self.gan_criterion = AdversarialLossFunction(
                use_least_squares=True,
                tensor_type=tensor_type
            )

    def setup_optimizers(self):
        """Initialize optimizers"""
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.G_lr,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.D_lr,
            betas=(0.5, 0.999)
        )

    def prepare_inputs(self, batch):
        """Prepare batch inputs for training"""
        # Cloth inputs
        cloth_paired = batch['cloth']['paired'].cuda()
        cloth_mask_paired = batch['cloth_mask']['paired'].cuda()
        cloth_mask_paired = torch.FloatTensor(
            (cloth_mask_paired.detach().cpu().numpy() > 0.5).astype(np.float32)
        ).cuda()

        # Pose inputs
        parse_agnostic = batch['parse_agnostic'].cuda()
        densepose = batch['densepose'].cuda()

        # Ground truth
        label_onehot = batch['parse_onehot'].cuda()
        label_segmentation = batch['parse'].cuda()
        parse_cloth_mask = batch['pcm'].cuda()
        target_cloth_region = batch['parse_cloth'].cuda()

        # Visualization
        person_image = batch['image']

        # Concatenate inputs
        cloth_input = torch.cat([cloth_paired, cloth_mask_paired], dim=1)
        pose_input = torch.cat([parse_agnostic, densepose], dim=1)

        return {
            'cloth_input': cloth_input,
            'pose_input': pose_input,
            'cloth_paired': cloth_paired,
            'cloth_mask_paired': cloth_mask_paired,
            'label_onehot': label_onehot,
            'label_segmentation': label_segmentation,
            'parse_cloth_mask': parse_cloth_mask,
            'target_cloth_region': target_cloth_region,
            'person_image': person_image
        }

    def forward_generator(self, inputs):
        """Forward pass through generator"""
        flow_list, fake_segmap, warped_cloth, warped_cloth_mask = self.generator(
            self.config,
            inputs['cloth_input'],
            inputs['pose_input']
        )

        # Binarize warped cloth mask
        warped_mask_binary = torch.FloatTensor(
            (warped_cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float32)
        ).cuda()

        # Apply cloth mask composition
        if self.config.clothmask_composition != 'no_composition':
            cloth_mask = torch.ones_like(fake_segmap.detach())

            if self.config.clothmask_composition == 'detach':
                cloth_mask[:, 3:4, :, :] = warped_mask_binary
            elif self.config.clothmask_composition == 'warp_grad':
                cloth_mask[:, 3:4, :, :] = warped_cloth_mask

            fake_segmap = fake_segmap * cloth_mask

        # Handle occlusions
        if self.config.occlusion:
            warped_cloth_mask = remove_occlusion_overlap(
                F.softmax(fake_segmap, dim=1),
                warped_cloth_mask
            )
            warped_cloth = (warped_cloth * warped_cloth_mask +
                            torch.ones_like(warped_cloth) * (1 - warped_cloth_mask))

        return {
            'flow_list': flow_list,
            'fake_segmap': fake_segmap,
            'warped_cloth': warped_cloth,
            'warped_cloth_mask': warped_cloth_mask,
            'warped_mask_binary': warped_mask_binary
        }

    def compute_generator_losses(self, inputs, outputs):
        """Compute all generator losses"""
        # Warping losses
        loss_l1_cloth = self.l1_criterion(
            outputs['warped_cloth_mask'],
            inputs['parse_cloth_mask']
        )
        loss_vgg = self.vgg_criterion(
            outputs['warped_cloth'],
            inputs['target_cloth_region']
        )

        # Total variation loss
        loss_tv = compute_total_variation_loss(
            outputs['flow_list'],
            self.config,
            outputs['warped_cloth_mask']
        )

        # Intermediate flow losses
        if self.config.intermediate_flow_loss:
            l1_intermediate, vgg_intermediate = compute_intermediate_flow_losses(
                outputs['flow_list'],
                inputs['cloth_paired'],
                inputs['cloth_mask_paired'],
                inputs['target_cloth_region'],
                outputs['fake_segmap'],
                self.config,
                self.l1_criterion,
                self.vgg_criterion
            )
            loss_l1_cloth += l1_intermediate
            loss_vgg += vgg_intermediate

        # Segmentation loss (Cross Entropy)
        loss_ce = cross_entropy2d(
            outputs['fake_segmap'],
            inputs['label_onehot'].transpose(0, 1)[0].long()
        )

        losses = {
            'l1_cloth': loss_l1_cloth,
            'vgg': loss_vgg,
            'tv': loss_tv,
            'ce': loss_ce
        }

        # GAN loss if enabled
        if not self.config.no_GAN_loss:
            fake_segmap_softmax = torch.softmax(outputs['fake_segmap'], dim=1)
            discriminator_pred = self.discriminator(torch.cat([
                inputs['cloth_input'].detach(),
                inputs['pose_input'].detach(),
                fake_segmap_softmax
            ], dim=1))

            losses['gan'] = self.gan_criterion(discriminator_pred, True)

        return losses

    def compute_discriminator_losses(self, inputs, outputs):
        """Compute discriminator losses"""
        fake_segmap_softmax = torch.softmax(
            outputs['fake_segmap'].detach(),
            dim=1
        )

        # Discriminator on fake
        fake_pred = self.discriminator(torch.cat([
            inputs['cloth_input'].detach(),
            inputs['pose_input'].detach(),
            fake_segmap_softmax
        ], dim=1))

        # Discriminator on real
        real_pred = self.discriminator(torch.cat([
            inputs['cloth_input'].detach(),
            inputs['pose_input'].detach(),
            inputs['label_segmentation']
        ], dim=1))

        loss_fake = self.gan_criterion(fake_pred, False)
        loss_real = self.gan_criterion(real_pred, True)

        return {
            'fake': loss_fake,
            'real': loss_real,
            'total': loss_fake + loss_real
        }

    def train_step(self, batch):
        """Single training step"""
        # Prepare inputs
        inputs = self.prepare_inputs(batch)

        # Forward generator
        outputs = self.forward_generator(inputs)

        # Compute generator losses
        g_losses = self.compute_generator_losses(inputs, outputs)

        # Total generator loss
        loss_G = (
                10 * g_losses['l1_cloth'] +
                g_losses['vgg'] +
                self.config.tv_lambda * g_losses['tv'] +
                self.config.CE_lambda * g_losses['ce']
        )

        if not self.config.no_GAN_loss:
            loss_G += self.config.GAN_lambda * g_losses['gan']

        # Update generator
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        # Update discriminator
        d_losses = None
        if not self.config.no_GAN_loss:
            if self.config.train_separately:
                # Re-generate for discriminator training
                with torch.no_grad():
                    outputs = self.forward_generator(inputs)

            d_losses = self.compute_discriminator_losses(inputs, outputs)

            self.optimizer_D.zero_grad()
            d_losses['total'].backward()
            self.optimizer_D.step()

        return {
            'g_losses': g_losses,
            'd_losses': d_losses,
            'loss_G': loss_G,
            'inputs': inputs,
            'outputs': outputs
        }

    def validate(self, val_loader):
        """Run validation"""
        self.generator.eval()
        iou_scores = []

        with torch.no_grad():
            for _ in range(2000 // self.config.batch_size):
                batch = val_loader.next_batch()
                inputs = self.prepare_inputs(batch)
                outputs = self.forward_generator(inputs)

                # Compute IoU
                iou = compute_iou_metric(
                    F.softmax(outputs['fake_segmap'], dim=1).detach(),
                    inputs['label_segmentation']
                )
                iou_scores.append(iou.item())

        self.generator.train()
        return np.mean(iou_scores)

    def train(self, train_loader, val_loader, test_loader, tensorboard_writer):
        """Main training loop"""
        for step in tqdm(range(self.config.load_step, self.config.keep_step)):
            iter_start_time = time.time()

            # Training step
            batch = train_loader.next_batch()
            step_results = self.train_step(batch)

            # Validation
            if (step + 1) % self.config.val_count == 0:
                mean_iou = self.validate(val_loader)
                tensorboard_writer.add_scalar('val/iou', mean_iou, step + 1)

            # TensorBoard logging
            if (step + 1) % self.config.tensorboard_count == 0:
                self.log_tensorboard(tensorboard_writer, step_results, step + 1)

                if not self.config.no_test_visualize:
                    self.visualize_test(test_loader, tensorboard_writer, step + 1)

            # Console logging
            if (step + 1) % self.config.display_count == 0:
                self.log_console(step_results, step + 1, iter_start_time)

            # Save checkpoint
            if (step + 1) % self.config.save_count == 0:
                self.save_checkpoint(step + 1)

    def log_tensorboard(self, writer, results, step):
        """Log metrics to TensorBoard"""
        g_losses = results['g_losses']
        d_losses = results['d_losses']

        # Generator losses
        writer.add_scalar('Loss/G/total', results['loss_G'].item(), step)
        writer.add_scalar('Loss/G/l1_cloth', g_losses['l1_cloth'].item(), step)
        writer.add_scalar('Loss/G/vgg', g_losses['vgg'].item(), step)
        writer.add_scalar('Loss/G/tv', g_losses['tv'].item(), step)
        writer.add_scalar('Loss/G/ce', g_losses['ce'].item(), step)

        if 'gan' in g_losses:
            writer.add_scalar('Loss/G/gan', g_losses['gan'].item(), step)

        # Discriminator losses
        if d_losses is not None:
            writer.add_scalar('Loss/D/total', d_losses['total'].item(), step)
            writer.add_scalar('Loss/D/real', d_losses['real'].item(), step)
            writer.add_scalar('Loss/D/fake', d_losses['fake'].item(), step)

        # Visualize training samples
        self.visualize_training(writer, results, step)

    def visualize_training(self, writer, results, step):
        """Create training visualization grid"""
        inputs = results['inputs']
        outputs = results['outputs']

        # Compute misalignment
        fake_cloth_mask = (torch.argmax(
            outputs['fake_segmap'].detach(), dim=1, keepdim=True
        ) == 3).long()
        misalignment = (fake_cloth_mask != inputs['parse_cloth_mask'].long()).float()

        # Create visualization grid
        vis_list = [
            inputs['cloth_paired'],
            inputs['pose_input'][:, :3, :, :],  # visualize first 3 pose channels
            outputs['warped_cloth'],
            torch.softmax(outputs['fake_segmap'], dim=1).argmax(1, keepdim=True).float() / inputs[
                'label_segmentation'].max(),
            inputs['person_image'],
            misalignment
        ]

        vis_list = [F.interpolate(v, size=(256, 192), mode='bilinear') for v in vis_list]
        grid = make_grid(torch.cat(vis_list, dim=0), nrow=len(vis_list), normalize=True, scale_each=True)
        writer.add_image('train/visualization', grid, step)

    def visualize_test(self, test_loader, writer, step):
        """Visualize test predictions"""
        self.generator.eval()
        test_batch = test_loader.next_batch()
        inputs = self.prepare_inputs(test_batch)

        with torch.no_grad():
            outputs = self.forward_generator(inputs)

        # Convert segmentation output to RGB for visualization
        fake_segmap = torch.softmax(outputs['fake_segmap'], dim=1).argmax(1, keepdim=True).float()
        vis_list = [
            inputs['cloth_paired'],
            outputs['warped_cloth'],
            fake_segmap / fake_segmap.max(),
            inputs['person_image']
        ]

        vis_list = [F.interpolate(v, size=(256, 192), mode='bilinear') for v in vis_list]
        grid = make_grid(torch.cat(vis_list, dim=0), nrow=len(vis_list), normalize=True, scale_each=True)
        writer.add_image('test/visualization', grid, step)

        self.generator.train()

    def log_console(self, results, step, start_time):
        """Print training status to console"""
        g_losses = results['g_losses']
        d_losses = results['d_losses']

        elapsed = time.time() - start_time
        log_str = f"[Step {step}] Time: {elapsed:.2f}s | " \
                  f"L1: {g_losses['l1_cloth']:.4f} | VGG: {g_losses['vgg']:.4f} | " \
                  f"TV: {g_losses['tv']:.4f} | CE: {g_losses['ce']:.4f}"

        if 'gan' in g_losses:
            log_str += f" | GAN: {g_losses['gan']:.4f}"

        if d_losses is not None:
            log_str += f" | D_total: {d_losses['total']:.4f} (real {d_losses['real']:.4f} / fake {d_losses['fake']:.4f})"

        print(log_str)

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{self.config.name}_step_{step}.pth")
        save_model_checkpoint({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'step': step
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def main():
    # Parse config
    config = TrainingConfig.parse_arguments()

    # Setup CUDA
    if torch.cuda.is_available() and config.gpu_ids != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
        config.cuda = True
        print(f"Using GPU: {config.gpu_ids}")
    else:
        config.cuda = False
        print("Using CPU")

    # Create necessary directories
    os.makedirs(config.tensorboard_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(logdir=os.path.join(config.tensorboard_dir, config.name))

    # Datasets and loaders
    train_dataset = VirtualTryOnDataset(config)
    val_dataset = VirtualTryOnDataset(config)
    test_dataset = VirtualTryOnDataset(config)

    train_loader = VirtualTryOnDataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                workers=config.workers)
    val_loader = VirtualTryOnDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, workers=config.workers)
    test_loader = VirtualTryOnDataLoader(test_dataset, batch_size=config.num_test_visualize, shuffle=False, workers=0)

    # Trainer
    trainer = VirtualTryOnTrainer(config)

    # Load checkpoint if specified
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        print(f"Loading checkpoint from {config.checkpoint_path}")
        checkpoint = load_model_checkpoint(config.checkpoint_path)
        trainer.generator.load_state_dict(checkpoint['generator'])
        trainer.discriminator.load_state_dict(checkpoint['discriminator'])
        trainer.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        trainer.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        config.load_step = checkpoint.get('step', 0)

    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, test_loader, writer)

if __name__ == "__main__":
    main()