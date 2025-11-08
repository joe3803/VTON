import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from VTON_Networks import create_sampling_grid as create_grid
import argparse
import os
import time
import numpy as np
from dataset import VirtualTryOnDataset, VirtualTryOnDataLoader
from VTON_Networks import WarpingFlowGenerator, PerceptualLoss, load_model_checkpoint,  save_model_checkpoint, create_sampling_grid
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import Subset


def calculate_iou(predictions, targets):
    batch_size = predictions.shape[0]
    total_iou = 0

    for idx in range(batch_size):
        pred = predictions[idx]
        target = targets[idx]

        pred_binary = pred > 0.5
        pred_flat = pred_binary.flatten()
        target_flat = target.flatten()

        intersection = torch.sum(pred_flat[target_flat == 1])
        union = torch.sum(pred_flat) + torch.sum(target_flat)

        total_iou += (intersection + 1e-7) / (union - intersection + 1e-7) / batch_size

    return total_iou


def handle_occlusion(segmentation_output, warped_cloth_mask):
    assert len(warped_cloth_mask.shape) == 4

    excluded_channels = torch.cat([
        segmentation_output[:, 1:3, :, :],
        segmentation_output[:, 5:, :, :]
    ], dim=1)

    overlap_mask = excluded_channels.sum(dim=1, keepdim=True)
    adjusted_mask = warped_cloth_mask - (overlap_mask * warped_cloth_mask)

    return adjusted_mask


class ConfigParser:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()

        parser.add_argument("--name", default="test")
        parser.add_argument("--gpu_ids", default="")
        parser.add_argument('-j', '--workers', type=int, default=4)
        parser.add_argument('-b', '--batch-size', type=int, default=8)
        parser.add_argument('--fp16', action='store_true', help='use amp')

        parser.add_argument("--dataroot", default="./data/")
        parser.add_argument("--datamode", default="train")
        parser.add_argument("--data_list", default="train_pairs.txt")
        parser.add_argument("--fine_width", type=int, default=192)
        parser.add_argument("--fine_height", type=int, default=256)

        parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--tocg_checkpoint', type=str, default='')

        parser.add_argument("--tensorboard_count", type=int, default=100)
        parser.add_argument("--display_count", type=int, default=100)
        parser.add_argument("--save_count", type=int, default=10000)
        parser.add_argument("--load_step", type=int, default=0)
        parser.add_argument("--keep_step", type=int, default=300000)
        parser.add_argument("--shuffle", action='store_true')
        parser.add_argument("--semantic_nc", type=int, default=13)
        parser.add_argument("--output_nc", type=int, default=13)

        parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
        parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
        parser.add_argument('--Ddownx2', action='store_true')
        parser.add_argument('--Ddropout', action='store_true')
        parser.add_argument('--num_D', type=int, default=2)
        parser.add_argument('--cuda', default=False)

        parser.add_argument("--G_D_seperate", action='store_true')
        parser.add_argument("--no_GAN_loss", action='store_true')
        parser.add_argument("--lasttvonly", action='store_true')
        parser.add_argument("--interflowloss", action='store_true')
        parser.add_argument("--clothmask_composition", type=str,
                            choices=['no_composition', 'detach', 'warp_grad'],
                            default='warp_grad')
        parser.add_argument('--edgeawaretv', type=str,
                            choices=['no_edge', 'last_only', 'weighted'],
                            default="no_edge")
        parser.add_argument('--add_lasttv', action='store_true')

        parser.add_argument("--no_test_visualize", action='store_true')
        parser.add_argument("--num_test_visualize", type=int, default=3)
        parser.add_argument("--test_datasetting", default="unpaired")
        parser.add_argument("--test_dataroot", default="./data/")
        parser.add_argument("--test_data_list", default="test_pairs.txt")

        parser.add_argument('--G_lr', type=float, default=0.0002)
        parser.add_argument('--D_lr', type=float, default=0.0002)
        parser.add_argument('--CElamda', type=float, default=10)
        parser.add_argument('--GANlambda', type=float, default=1)
        parser.add_argument('--tvlambda', type=float, default=2)
        parser.add_argument('--upsample', type=str, default='bilinear',
                            choices=['nearest', 'bilinear'])
        parser.add_argument('--val_count', type=int, default='1000')
        parser.add_argument('--spectral', action='store_true')
        parser.add_argument('--occlusion', action='store_true')

        return parser.parse_args()


class LossCalculator:
    def __init__(self, config):
        self.config = config
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = PerceptualLoss(config)

        if config.fp16:
            self.gan_loss = PerceptualLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
        else:
            tensor_type = torch.cuda.FloatTensor if config.gpu_ids else torch.Tensor
            self.gan_loss = PerceptualLoss(use_lsgan=True, tensor=tensor_type)

    def compute_tv_loss(self, flow_list, warped_mask=None):
        total_tv_loss = 0

        if self.config.edgeawaretv == 'no_edge':
            flows = flow_list if not self.config.lasttvonly else flow_list[-1:]

            for flow in flows:
                y_variation = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_variation = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                total_tv_loss += y_variation + x_variation

        elif self.config.edgeawaretv == 'last_only':
            total_tv_loss = self._compute_edge_aware_tv(flow_list[-1], warped_mask)

        elif self.config.edgeawaretv == 'weighted':
            for idx in range(5):
                flow = flow_list[idx]
                weight = 1.0 / (2 ** (4 - idx))
                total_tv_loss += self._compute_edge_aware_tv(flow, warped_mask, weight)

        if self.config.add_lasttv:
            flow = flow_list[-1]
            y_variation = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
            x_variation = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
            total_tv_loss += y_variation + x_variation

        return total_tv_loss

    def _compute_edge_aware_tv(self, flow, warped_mask, weight=1.0):
        downsampled_mask = F.interpolate(warped_mask, flow.shape[1:3], mode='bilinear')

        y_variation = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
        x_variation = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

        mask_permuted = downsampled_mask.permute(0, 2, 3, 1)
        y_edge_weight = torch.exp(-150 * torch.abs(mask_permuted[:, 1:, :, :] - mask_permuted[:, :-1, :, :]))
        x_edge_weight = torch.exp(-150 * torch.abs(mask_permuted[:, :, 1:, :] - mask_permuted[:, :, :-1, :]))

        weighted_y = (y_variation * y_edge_weight).mean() * weight
        weighted_x = (x_variation * x_edge_weight).mean() * weight

        return weighted_y + weighted_x


class Trainer:
    def __init__(self, config, generator, discriminator, train_loader, val_loader, test_loader, logger):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger

        self.loss_calculator = LossCalculator(config)

        self.optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=config.G_lr,
            betas=(0.5, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.D_lr,
            betas=(0.5, 0.999)
        )

        self.generator.cuda()
        self.generator.train()
        self.discriminator.cuda()
        self.discriminator.train()

    def prepare_inputs(self, batch_data):
        cloth_rgb = batch_data['cloth']['paired'].cuda()
        cloth_mask = batch_data['cloth_mask']['paired'].cuda()
        cloth_mask = torch.FloatTensor((cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()

        parse_agnostic = batch_data['parse_agnostic'].cuda()
        densepose = batch_data['densepose'].cuda()

        label_onehot = batch_data['parse_onehot'].cuda()
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
            'label_onehot': label_onehot,
            'label_segmap': label_segmap,
            'parse_cloth_mask': parse_cloth_mask,
            'parse_cloth_rgb': parse_cloth_rgb,
            'image': batch_data['image']
        }

    def forward_generator(self, inputs):
        flow_list, pred_segmap, warped_cloth, warped_mask = self.generator(
            inputs['input_cloth'],
            inputs['input_body']
        )

        warped_mask_binary = torch.FloatTensor(
            (warped_mask.detach().cpu().numpy() > 0.5).astype(np.float32)
        ).cuda()

        if self.config.clothmask_composition != 'no_composition':
            cloth_channel_mask = torch.ones_like(pred_segmap.detach())

            if self.config.clothmask_composition == 'detach':
                cloth_channel_mask[:, 3:4, :, :] = warped_mask_binary
            elif self.config.clothmask_composition == 'warp_grad':
                cloth_channel_mask[:, 3:4, :, :] = warped_mask

            pred_segmap = pred_segmap * cloth_channel_mask

        if self.config.occlusion:
            warped_mask = handle_occlusion(F.softmax(pred_segmap, dim=1), warped_mask)
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)

        pred_cloth_channel = (torch.argmax(pred_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalignment = pred_cloth_channel - warped_mask_binary
        misalignment[misalignment < 0.0] = 0.0

        return {
            'flow_list': flow_list,
            'pred_segmap': pred_segmap,
            'warped_cloth': warped_cloth,
            'warped_mask': warped_mask,
            'warped_mask_binary': warped_mask_binary,
            'misalignment': misalignment
        }

    def compute_warping_losses(self, outputs, inputs):
        loss_cloth_mask = self.loss_calculator.l1_loss(
            outputs['warped_mask'],
            inputs['parse_cloth_mask']
        )

        loss_vgg = self.loss_calculator.vgg_loss(
            outputs['warped_cloth'],
            inputs['parse_cloth_rgb']
        )

        loss_tv = self.loss_calculator.compute_tv_loss(
            outputs['flow_list'],
            outputs['warped_mask']
        )

        if self.config.interflowloss:
            cloth_rgb = inputs['cloth_rgb']
            cloth_mask = inputs['cloth_mask']
            batch_size, _, img_height, img_width = cloth_rgb.size()

            for idx in range(len(outputs['flow_list']) - 1):
                flow = outputs['flow_list'][idx]
                _, flow_h, flow_w, _ = flow.size()

                grid = create_grid(batch_size, img_height, img_width)
                upsampled_flow = F.interpolate(
                    flow.permute(0, 3, 1, 2),
                    size=cloth_rgb.shape[2:],
                    mode=self.config.upsample
                ).permute(0, 2, 3, 1)

                normalized_flow = torch.cat([
                    upsampled_flow[:, :, :, 0:1] / ((flow_w - 1.0) / 2.0),
                    upsampled_flow[:, :, :, 1:2] / ((flow_h - 1.0) / 2.0)
                ], 3)

                warped_cloth_intermediate = F.grid_sample(
                    cloth_rgb,
                    normalized_flow + grid,
                    padding_mode='border'
                )
                warped_mask_intermediate = F.grid_sample(
                    cloth_mask,
                    normalized_flow + grid,
                    padding_mode='border'
                )

                warped_mask_intermediate = handle_occlusion(
                    F.softmax(outputs['pred_segmap'], dim=1),
                    warped_mask_intermediate
                )

                scale_weight = 1.0 / (2 ** (4 - idx))
                loss_cloth_mask += self.loss_calculator.l1_loss(
                    warped_mask_intermediate,
                    inputs['parse_cloth_mask']
                ) * scale_weight

                loss_vgg += self.loss_calculator.vgg_loss(
                    warped_cloth_intermediate,
                    inputs['parse_cloth_rgb']
                ) * scale_weight

        return loss_cloth_mask, loss_vgg, loss_tv

    def train_step(self, step):
        start_time = time.time()
        batch_data = self.train_loader.next_batch()

        inputs = self.prepare_inputs(batch_data)
        outputs = self.forward_generator(inputs)

        loss_cloth_mask, loss_vgg, loss_tv = self.compute_warping_losses(outputs, inputs)

        ce_loss = cross_entropy2d(
            outputs['pred_segmap'],
            inputs['label_onehot'].transpose(0, 1)[0].long()
        )

        if self.config.no_GAN_loss:
            total_loss_g = (
                    10 * loss_cloth_mask +
                    loss_vgg +
                    self.config.tvlambda * loss_tv +
                    self.config.CElamda * ce_loss
            )

            self.optimizer_g.zero_grad()
            total_loss_g.backward()
            self.optimizer_g.step()

            losses = {
                'loss_g': total_loss_g,
                'loss_cloth_mask': loss_cloth_mask,
                'loss_vgg': loss_vgg,
                'loss_tv': loss_tv,
                'ce_loss': ce_loss
            }
        else:
            pred_segmap_softmax = torch.softmax(outputs['pred_segmap'], 1)

            discriminator_input_fake = torch.cat((
                inputs['input_cloth'].detach(),
                inputs['input_body'].detach(),
                pred_segmap_softmax
            ), dim=1)

            pred_fake_for_g = self.discriminator(discriminator_input_fake)
            loss_g_gan = self.loss_calculator.gan_loss(pred_fake_for_g, True)

            if not self.config.G_D_seperate:
                discriminator_input_fake_detached = torch.cat((
                    inputs['input_cloth'].detach(),
                    inputs['input_body'].detach(),
                    pred_segmap_softmax.detach()
                ), dim=1)

                discriminator_input_real = torch.cat((
                    inputs['input_cloth'].detach(),
                    inputs['input_body'].detach(),
                    inputs['label_segmap']
                ), dim=1)

                pred_fake_for_d = self.discriminator(discriminator_input_fake_detached)
                pred_real_for_d = self.discriminator(discriminator_input_real)

                loss_d_fake = self.loss_calculator.gan_loss(pred_fake_for_d, False)
                loss_d_real = self.loss_calculator.gan_loss(pred_real_for_d, True)

                total_loss_g = (
                        10 * loss_cloth_mask +
                        loss_vgg +
                        self.config.tvlambda * loss_tv +
                        self.config.CElamda * ce_loss +
                        self.config.GANlambda * loss_g_gan
                )

                total_loss_d = loss_d_fake + loss_d_real

                self.optimizer_g.zero_grad()
                total_loss_g.backward()
                self.optimizer_g.step()

                self.optimizer_d.zero_grad()
                total_loss_d.backward()
                self.optimizer_d.step()

            else:
                total_loss_g = (
                        10 * loss_cloth_mask +
                        loss_vgg +
                        self.config.tvlambda * loss_tv +
                        self.config.CElamda * ce_loss +
                        self.config.GANlambda * loss_g_gan
                )

                self.optimizer_g.zero_grad()
                total_loss_g.backward()
                self.optimizer_g.step()

                with torch.no_grad():
                    _, fresh_pred_segmap, _, _ = self.generator(
                        inputs['input_cloth'],
                        inputs['input_body']
                    )

                fresh_pred_softmax = torch.softmax(fresh_pred_segmap, 1)

                discriminator_input_fake_fresh = torch.cat((
                    inputs['input_cloth'].detach(),
                    inputs['input_body'].detach(),
                    fresh_pred_softmax.detach()
                ), dim=1)

                discriminator_input_real = torch.cat((
                    inputs['input_cloth'].detach(),
                    inputs['input_body'].detach(),
                    inputs['label_segmap']
                ), dim=1)

                pred_fake_for_d = self.discriminator(discriminator_input_fake_fresh)
                pred_real_for_d = self.discriminator(discriminator_input_real)

                loss_d_fake = self.loss_calculator.gan_loss(pred_fake_for_d, False)
                loss_d_real = self.loss_calculator.gan_loss(pred_real_for_d, True)

                total_loss_d = loss_d_fake + loss_d_real

                self.optimizer_d.zero_grad()
                total_loss_d.backward()
                self.optimizer_d.step()

            losses = {
                'loss_g': total_loss_g,
                'loss_cloth_mask': loss_cloth_mask,
                'loss_vgg': loss_vgg,
                'loss_tv': loss_tv,
                'ce_loss': ce_loss,
                'loss_g_gan': loss_g_gan,
                'loss_d': total_loss_d,
                'loss_d_real': loss_d_real,
                'loss_d_fake': loss_d_fake
            }

        elapsed_time = time.time() - start_time

        return losses, outputs, inputs, elapsed_time

    def validate(self, step):
        self.generator.eval()
        iou_scores = []

        with torch.no_grad():
            for _ in range(2000 // self.config.batch_size):
                batch_data = self.val_loader.next_batch()
                inputs = self.prepare_inputs(batch_data)
                outputs = self.forward_generator(inputs)

                iou = calculate_iou(
                    F.softmax(outputs['pred_segmap'], dim=1).detach(),
                    inputs['label_segmap']
                )
                iou_scores.append(iou.item())

        self.generator.train()
        self.logger.add_scalar('val/iou', np.mean(iou_scores), step + 1)

    def log_training(self, step, losses, outputs, inputs):
        self.logger.add_scalar('Loss/G', losses['loss_g'].item(), step + 1)
        self.logger.add_scalar('Loss/G/l1_cloth', losses['loss_cloth_mask'].item(), step + 1)
        self.logger.add_scalar('Loss/G/vgg', losses['loss_vgg'].item(), step + 1)
        self.logger.add_scalar('Loss/G/tv', losses['loss_tv'].item(), step + 1)
        self.logger.add_scalar('Loss/G/CE', losses['ce_loss'].item(), step + 1)

        if not self.config.no_GAN_loss:
            self.logger.add_scalar('Loss/G/GAN', losses['loss_g_gan'].item(), step + 1)
            self.logger.add_scalar('Loss/D', losses['loss_d'].item(), step + 1)
            self.logger.add_scalar('Loss/D/pred_real', losses['loss_d_real'].item(), step + 1)
            self.logger.add_scalar('Loss/D/pred_fake', losses['loss_d_fake'].item(), step + 1)

        grid = make_grid([
            (inputs['cloth_rgb'][0].cpu() / 2 + 0.5),
            (inputs['cloth_mask'][0].cpu()).expand(3, -1, -1),
            visualize_segmap(inputs['input_body'][:, :13].cpu()),
            ((inputs['input_body'][:, 13:].cpu()[0] + 1) / 2),
            (inputs['parse_cloth_rgb'][0].cpu() / 2 + 0.5),
            inputs['parse_cloth_mask'][0].cpu().expand(3, -1, -1),
            (outputs['warped_cloth'][0].cpu().detach() / 2 + 0.5),
            (outputs['warped_mask_binary'][0].cpu().detach()).expand(3, -1, -1),
            visualize_segmap(inputs['label_segmap'].cpu()),
            visualize_segmap(outputs['pred_segmap'].cpu()),
            (inputs['image'][0] / 2 + 0.5),
            (outputs['misalignment'][0].cpu().detach()).expand(3, -1, -1)
        ], nrow=4)

        self.logger.add_images('train_images', grid.unsqueeze(0), step + 1)

    def visualize_test(self, step):
        if self.config.no_test_visualize:
            return

        batch_data = self.test_loader.next_batch()

        cloth_rgb = batch_data['cloth'][self.config.test_datasetting].cuda()
        cloth_mask = batch_data['cloth_mask'][self.config.test_datasetting].cuda()
        cloth_mask = torch.FloatTensor((cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()

        inputs = {
            'cloth_rgb': cloth_rgb,
            'cloth_mask': cloth_mask,
            'input_cloth': torch.cat([cloth_rgb, cloth_mask], 1),
            'input_body': torch.cat([
                batch_data['parse_agnostic'].cuda(),
                batch_data['densepose'].cuda()
            ], 1),
            'label_segmap': batch_data['parse'].cuda(),
            'parse_cloth_mask': batch_data['pcm'].cuda(),
            'parse_cloth_rgb': batch_data['parse_cloth'].cuda(),
            'image': batch_data['image']
        }

        self.generator.eval()

        with torch.no_grad():
            outputs = self.forward_generator(inputs)

        for idx in range(self.config.num_test_visualize):
            grid = make_grid([
                (cloth_rgb[idx].cpu() / 2 + 0.5),
                (cloth_mask[idx].cpu()).expand(3, -1, -1),
                visualize_segmap(inputs['input_body'][:, :13].cpu(), batch=idx),
                ((inputs['input_body'][:, 13:].cpu()[idx] + 1) / 2),
                (inputs['parse_cloth_rgb'][idx].cpu() / 2 + 0.5),
                inputs['parse_cloth_mask'][idx].cpu().expand(3, -1, -1),
                (outputs['warped_cloth'][idx].cpu().detach() / 2 + 0.5),
                (outputs['warped_mask_binary'][idx].cpu().detach()).expand(3, -1, -1),
                visualize_segmap(inputs['label_segmap'].cpu(), batch=idx),
                visualize_segmap(outputs['pred_segmap'].cpu(), batch=idx),
                (inputs['image'][idx] / 2 + 0.5),
                (outputs['misalignment'][idx].cpu().detach()).expand(3, -1, -1)
            ], nrow=4)

            self.logger.add_images(f'test_images/{idx}', grid.unsqueeze(0), step + 1)

        self.generator.train()

    def run(self):
        for step in tqdm(range(self.config.load_step, self.config.keep_step)):
            losses, outputs, inputs, elapsed_time = self.train_step(step)

            if (step + 1) % self.config.val_count == 0:
                self.validate(step)

            if (step + 1) % self.config.tensorboard_count == 0:
                self.log_training(step, losses, outputs, inputs)
                self.visualize_test(step)

            if (step + 1) % self.config.display_count == 0:
                if not self.config.no_GAN_loss:
                    print(
                        f"step: {step + 1:8d}, time: {elapsed_time:.3f}\n"
                        f"loss G: {losses['loss_g'].item():.4f}, "
                        f"L1_cloth: {losses['loss_cloth_mask'].item():.4f}, "
                        f"VGG: {losses['loss_vgg'].item():.4f}, "
                        f"TV: {losses['loss_tv'].item():.4f}, "
                        f"CE: {losses['ce_loss'].item():.4f}, "
                        f"G GAN: {losses['loss_g_gan'].item():.4f}\n"
                        f"loss D: {losses['loss_d'].item():.4f}, "
                        f"D real: {losses['loss_d_real'].item():.4f}, "
                        f"D fake: {losses['loss_d_fake'].item():.4f}",
                        flush=True
                    )

            if (step + 1) % self.config.save_count == 0:
                save_model_checkpoint(
                    self.generator,
                    os.path.join(self.config.checkpoint_dir, self.config.name,
                                 f'tocg_step_{step + 1:06d}.pth'),
                    self.config
                )
                save_model_checkpoint(
                    self.discriminator,
                    os.path.join(self.config.checkpoint_dir, self.config.name,
                                 f'D_step_{step + 1:06d}.pth'),
                    self.config
                )


def setup_environment(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids

    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)

    checkpoint_path = os.path.join(config.checkpoint_dir, config.name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


def create_data_loaders(config):
    train_dataset = VirtualTryOnDataset(config)
    train_loader = VirtualTryOnDataLoader(config, train_dataset)

    test_loader = None
    val_loader = None

    if not config.no_test_visualize:
        original_batch_size = config.batch_size
        original_dataroot = config.dataroot
        original_datamode = config.datamode
        original_data_list = config.data_list

        config.batch_size = config.num_test_visualize
        config.dataroot = config.test_dataroot
        config.datamode = 'test'
        config.data_list = config.test_data_list

        test_dataset = VirtualTryOnDataset(config)
        val_dataset = Subset(test_dataset, np.arange(2000))

        test_loader = VirtualTryOnDataLoader(config, test_dataset)
        val_loader = VirtualTryOnDataLoader(config, val_dataset)

        config.batch_size = original_batch_size
        config.dataroot = original_dataroot
        config.datamode = original_datamode
        config.data_list = original_data_list

    return train_loader, val_loader, test_loader


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

    discriminator = define_D(
        input_nc=cloth_input_channels + body_input_channels + config.output_nc,
        Ddownx2=config.Ddownx2,
        Ddropout=config.Ddropout,
        n_layers_D=3,
        spectral=config.spectral,
        num_D=config.num_D
    )

    if config.tocg_checkpoint and os.path.exists(config.tocg_checkpoint):
        print(f"Loading generator checkpoint from: {config.tocg_checkpoint}")
        load_model_checkpoint(generator, config.tocg_checkpoint)

    return generator, discriminator


def save_final_checkpoints(config, generator, discriminator):
    generator_path = os.path.join(config.checkpoint_dir, config.name, 'tocg_final.pth')
    discriminator_path = os.path.join(config.checkpoint_dir, config.name, 'D_final.pth')

    print(f"Saving final generator checkpoint to: {generator_path}")
    save_model_checkpoint(generator, generator_path, config)

    print(f"Saving final discriminator checkpoint to: {discriminator_path}")
    save_model_checkpoint(discriminator, discriminator_path, config)


def main():
    config = ConfigParser.parse_arguments()

    print("=" * 80)
    print(f"Training Configuration: {config.name}")
    print("=" * 80)
    print(config)
    print("=" * 80)

    setup_environment(config)

    print("\n[1/4] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"✓ Train loader created")
    if val_loader:
        print(f"✓ Validation loader created (2000 samples)")
    if test_loader:
        print(f"✓ Test loader created ({config.num_test_visualize} samples)")

    print("\n[2/4] Initializing models...")
    generator, discriminator = create_models(config)
    print(f"✓ Generator initialized")
    print(f"✓ Discriminator initialized")

    print("\n[3/4] Setting up logger...")
    logger = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.name))
    print(f"✓ TensorBoard logger created at: {os.path.join(config.tensorboard_dir, config.name)}")

    print("\n[4/4] Starting training...")
    print(f"Training steps: {config.load_step} → {config.keep_step}")
    print(f"Batch size: {config.batch_size}")
    print(f"Generator LR: {config.G_lr}")
    print(f"Discriminator LR: {config.D_lr}")
    print("=" * 80)

    trainer = Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger
    )

    trainer.run()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    save_final_checkpoints(config, generator, discriminator)

    logger.close()

    print(f"\n✓ All done! Model: {config.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()