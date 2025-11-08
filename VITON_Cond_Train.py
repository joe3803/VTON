import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid as make_image_grid
import torchgeometry as tgm

import argparse
import os
import time
import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import VirtualTryOnDataset, VirtualTryOnDataLoader
from dataset_test import VirtualTryOnDataset
from VTON_Networks import WarpingFlowGenerator, PerceptualLoss, load_model_checkpoint,  save_model_checkpoint, create_sampling_grid
from VTON_Networks_gen import SPADEGenerator, MultiscaleDiscriminator, compute_loss
from sync_batchnorm import DataParallelWithCallback
from tensorboardX import SummaryWriter
from utils import create_network, visualize_segmap
import eval_models as models


class TrainingConfig:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--name', type=str, required=True)
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('-j', '--workers', type=int, default=4)
        parser.add_argument('-b', '--batch_size', type=int, default=8)
        parser.add_argument('--fp16', action='store_true', help='use amp')
        parser.add_argument('--cuda', default=False, help='cuda or cpu')

        parser.add_argument("--dataroot", default="./data/")
        parser.add_argument("--datamode", default="train")
        parser.add_argument("--data_list", default="train_pairs.txt")
        parser.add_argument("--fine_width", type=int, default=768)
        parser.add_argument("--fine_height", type=int, default=1024)
        parser.add_argument("--radius", type=int, default=20)
        parser.add_argument("--grid_size", type=int, default=5)

        parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--tocg_checkpoint', type=str)
        parser.add_argument('--gen_checkpoint', type=str, default='')
        parser.add_argument('--dis_checkpoint', type=str, default='')

        parser.add_argument("--tensorboard_count", type=int, default=100)
        parser.add_argument("--display_count", type=int, default=100)
        parser.add_argument("--save_count", type=int, default=10000)
        parser.add_argument("--load_step", type=int, default=0)
        parser.add_argument("--keep_step", type=int, default=100000)
        parser.add_argument("--decay_step", type=int, default=100000)
        parser.add_argument("--shuffle", action='store_true')

        parser.add_argument("--lpips_count", type=int, default=1000)
        parser.add_argument("--test_datasetting", default="paired")
        parser.add_argument("--test_dataroot", default="./data/")
        parser.add_argument("--test_data_list", default="test_pairs.txt")

        parser.add_argument('--G_lr', type=float, default=0.0001)
        parser.add_argument('--D_lr', type=float, default=0.0004)

        parser.add_argument('--GMM_const', type=float, default=None)
        parser.add_argument('--semantic_nc', type=int, default=13)
        parser.add_argument('--gen_semantic_nc', type=int, default=7)
        parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
        parser.add_argument('--norm_D', type=str, default='spectralinstance')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most')
        parser.add_argument('--init_type', type=str, default='xavier')
        parser.add_argument('--init_variance', type=float, default=0.02)

        parser.add_argument('--no_ganFeat_loss', action='store_true')
        parser.add_argument('--no_vgg_loss', action='store_true')
        parser.add_argument('--lambda_l1', type=float, default=1.0)
        parser.add_argument('--lambda_feat', type=float, default=10.0)
        parser.add_argument('--lambda_vgg', type=float, default=10.0)

        parser.add_argument('--n_layers_D', type=int, default=3)
        parser.add_argument('--netD_subarch', type=str, default='n_layer')
        parser.add_argument('--num_D', type=int, default=2)

        parser.add_argument('--GT', action='store_true')
        parser.add_argument('--occlusion', action='store_true')
        parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
        parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
        parser.add_argument("--clothmask_composition", type=str,
                            choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        parser.add_argument("--num_test_visualize", type=int, default=3)

        self.config = parser.parse_args()
        self._setup_gpus()

    def _setup_gpus(self):
        gpu_list = self.config.gpu_ids.split(',')
        self.config.gpu_ids = [int(x) for x in gpu_list if int(x) >= 0]

        if len(self.config.gpu_ids) > 0:
            torch.cuda.set_device(self.config.gpu_ids[0])

        assert len(self.config.gpu_ids) == 0 or self.config.batch_size % len(self.config.gpu_ids) == 0, \
            f"Batch size {self.config.batch_size} must be multiple of GPUs {len(self.config.gpu_ids)}"

    def get(self):
        return self.config


class ClothWarper:
    def __init__(self, config):
        self.config = config
        self.gaussian_blur = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()

    def remove_overlap(self, segmentation, warped_mask):
        assert len(warped_mask.shape) == 4
        overlap = torch.cat([segmentation[:, 1:3, :, :], segmentation[:, 5:, :, :]], dim=1)
        overlap_sum = overlap.sum(dim=1, keepdim=True)
        return warped_mask - overlap_sum * warped_mask

    def warp_cloth(self, cloth, cloth_mask, parse_agnostic, densepose, tocg, use_occlusion=False):
        downsampled_mask = F.interpolate(cloth_mask, size=(256, 192), mode='nearest')
        downsampled_parse = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
        downsampled_cloth = F.interpolate(cloth, size=(256, 192), mode='bilinear')
        downsampled_pose = F.interpolate(densepose, size=(256, 192), mode='bilinear')

        cloth_input = torch.cat([downsampled_cloth, downsampled_mask], 1)
        context_input = torch.cat([downsampled_parse, downsampled_pose], 1)

        flow_list, segmentation, _, warped_mask = tocg(cloth_input, context_input)

        mask_binary = torch.FloatTensor(
            (warped_mask.detach().cpu().numpy() > 0.5).astype(np.float)
        ).cuda()

        if self.config.clothmask_composition != 'no_composition':
            mask_modifier = torch.ones_like(segmentation)

            if self.config.clothmask_composition == 'detach':
                mask_modifier[:, 3:4, :, :] = mask_binary
            elif self.config.clothmask_composition == 'warp_grad':
                mask_modifier[:, 3:4, :, :] = warped_mask

            segmentation = segmentation * mask_modifier

        batch_size, _, height, width = cloth.shape
        sampling_grid = create_sampling_grid(batch_size, height, width, self.config)

        flow = F.interpolate(
            flow_list[-1].permute(0, 3, 1, 2),
            size=(height, width),
            mode='bilinear'
        ).permute(0, 2, 3, 1)

        flow_normalized = torch.cat([
            flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0),
            flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)
        ], 3)

        warped_grid = sampling_grid + flow_normalized
        warped_cloth = F.grid_sample(cloth, warped_grid, padding_mode='border').detach()
        warped_mask_full = F.grid_sample(cloth_mask, warped_grid, padding_mode='border')

        blurred_segmentation = self.gaussian_blur(
            F.interpolate(segmentation, size=(height, width), mode='bilinear')
        )
        final_parse = blurred_segmentation.argmax(dim=1)[:, None]

        if use_occlusion:
            warped_mask_full = self.remove_overlap(
                F.softmax(blurred_segmentation, dim=1),
                warped_mask_full
            )
            warped_cloth = warped_cloth * warped_mask_full + \
                           torch.ones_like(warped_cloth) * (1 - warped_mask_full)
            warped_cloth = warped_cloth.detach()

        return warped_cloth, final_parse, blurred_segmentation

    def convert_parse_map(self, parse_indices):
        old_parse = torch.FloatTensor(
            parse_indices.size(0), 13,
            self.config.fine_height, self.config.fine_width
        ).zero_().cuda()
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

        new_parse = torch.FloatTensor(
            parse_indices.size(0), 7,
            self.config.fine_height, self.config.fine_width
        ).zero_().cuda()

        for idx in range(len(label_mapping)):
            for label in label_mapping[idx][1]:
                new_parse[:, idx] += old_parse[:, label]

        return new_parse.detach()


class GANTrainer:
    def __init__(self, config, generator, discriminator, warping_net=None):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.warping_net = warping_net
        self.cloth_warper = ClothWarper(config)

        self._setup_losses()
        self._setup_optimizers()
        self._setup_distributed()

    def _setup_losses(self):
        tensor_type = torch.cuda.HalfTensor if self.config.fp16 else torch.cuda.FloatTensor
        self.gan_loss = PerceptualLoss('hinge', tensor=tensor_type)
        self.feature_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(self.config)

    def _setup_optimizers(self):
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.G_lr,
            betas=(0, 0.9)
        )

        self.dis_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.D_lr,
            betas=(0, 0.9)
        )

        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.gen_optimizer,
            lr_lambda=lambda step: 1.0 - max(0, step * 1000 + self.config.load_step - self.config.keep_step) /
                                   float(self.config.decay_step + 1)
        )

        self.dis_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.dis_optimizer,
            lr_lambda=lambda step: 1.0 - max(0, step * 1000 + self.config.load_step - self.config.keep_step) /
                                   float(self.config.decay_step + 1)
        )

    def _setup_distributed(self):
        if self.config.fp16:
            from apex import amp
            models_to_amp = [self.generator, self.discriminator]
            optimizers = [self.gen_optimizer, self.dis_optimizer]

            if not self.config.GT:
                models_to_amp.insert(0, self.warping_net)

            models_to_amp, optimizers = amp.initialize(
                models_to_amp, optimizers, opt_level='O1', num_losses=2
            )

        if len(self.config.gpu_ids) > 0:
            if not self.config.GT:
                self.warping_net = DataParallelWithCallback(
                    self.warping_net, device_ids=self.config.gpu_ids
                )
            self.generator = DataParallelWithCallback(
                self.generator, device_ids=self.config.gpu_ids
            )
            self.discriminator = DataParallelWithCallback(
                self.discriminator, device_ids=self.config.gpu_ids
            )
            self.gan_loss = DataParallelWithCallback(
                self.gan_loss, device_ids=self.config.gpu_ids
            )
            self.feature_loss = DataParallelWithCallback(
                self.feature_loss, device_ids=self.config.gpu_ids
            )
            self.perceptual_loss = DataParallelWithCallback(
                self.perceptual_loss, device_ids=self.config.gpu_ids
            )

    def prepare_inputs(self, batch_data, use_gt=False, paired=True):
        person_representation = batch_data['agnostic'].cuda()
        ground_truth_parse = batch_data['parse'].cuda()
        densepose = batch_data['densepose'].cuda()
        cloth_segmentation = batch_data['parse_cloth'].cuda()
        agnostic_parse = batch_data['parse_agnostic'].cuda()

        mode = 'paired' if paired else 'unpaired'
        cloth_mask = batch_data['cloth_mask'][mode].cuda()
        cloth_image = batch_data['cloth'][mode].cuda()
        target_image = batch_data['image'].cuda()

        if use_gt:
            warped_cloth = cloth_segmentation
            parse_indices = ground_truth_parse.argmax(dim=1)[:, None]
        else:
            warped_cloth, parse_indices, segmentation = self.cloth_warper.warp_cloth(
                cloth_image, cloth_mask, agnostic_parse, densepose,
                self.warping_net, self.config.occlusion
            )

        parse_map = self.cloth_warper.convert_parse_map(parse_indices)

        return {
            'person_rep': person_representation,
            'densepose': densepose,
            'warped_cloth': warped_cloth,
            'parse_map': parse_map,
            'target': target_image,
            'cloth': cloth_image,
            'cloth_mask': cloth_mask,
            'agnostic_parse': agnostic_parse,
            'segmentation': segmentation if not use_gt else None
        }

    def train_generator_step(self, prepared_data):
        generated_image = self.generator(
            torch.cat((
                prepared_data['person_rep'],
                prepared_data['densepose'],
                prepared_data['warped_cloth']
            ), dim=1),
            prepared_data['parse_map']
        )

        fake_input = torch.cat((prepared_data['parse_map'], generated_image), dim=1)
        real_input = torch.cat((prepared_data['parse_map'], prepared_data['target']), dim=1)
        discriminator_output = self.discriminator(torch.cat((fake_input, real_input), dim=0))

        if isinstance(discriminator_output, list):
            fake_predictions = []
            real_predictions = []
            for pred in discriminator_output:
                fake_predictions.append([t[:t.size(0) // 2] for t in pred])
                real_predictions.append([t[t.size(0) // 2:] for t in pred])
        else:
            fake_predictions = discriminator_output[:discriminator_output.size(0) // 2]
            real_predictions = discriminator_output[discriminator_output.size(0) // 2:]

        losses = {}
        losses['adversarial'] = self.gan_loss(fake_predictions, True, for_discriminator=False)

        if not self.config.no_ganFeat_loss:
            num_discriminators = len(fake_predictions)
            feature_loss = torch.cuda.FloatTensor(len(self.config.gpu_ids)).zero_()

            for i in range(num_discriminators):
                num_layers = len(fake_predictions[i]) - 1
                for j in range(num_layers):
                    layer_loss = self.feature_loss(
                        fake_predictions[i][j],
                        real_predictions[i][j].detach()
                    )
                    feature_loss += layer_loss * self.config.lambda_feat / num_discriminators

            losses['feature_matching'] = feature_loss

        if not self.config.no_vgg_loss:
            losses['perceptual'] = self.perceptual_loss(
                generated_image,
                prepared_data['target']
            ) * self.config.lambda_vgg

        total_loss = sum(losses.values()).mean()

        self.gen_optimizer.zero_grad()
        if self.config.fp16:
            from apex import amp
            with amp.scale_loss(total_loss, self.gen_optimizer, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        self.gen_optimizer.step()

        return total_loss, losses, generated_image

    def train_discriminator_step(self, prepared_data):
        with torch.no_grad():
            generated_image = self.generator(
                torch.cat((
                    prepared_data['person_rep'],
                    prepared_data['densepose'],
                    prepared_data['warped_cloth']
                ), dim=1),
                prepared_data['parse_map']
            )
            generated_image = generated_image.detach()
            generated_image.requires_grad_()

        fake_input = torch.cat((prepared_data['parse_map'], generated_image), dim=1)
        real_input = torch.cat((prepared_data['parse_map'], prepared_data['target']), dim=1)
        discriminator_output = self.discriminator(torch.cat((fake_input, real_input), dim=0))

        if isinstance(discriminator_output, list):
            fake_predictions = []
            real_predictions = []
            for pred in discriminator_output:
                fake_predictions.append([t[:t.size(0) // 2] for t in pred])
                real_predictions.append([t[t.size(0) // 2:] for t in pred])
        else:
            fake_predictions = discriminator_output[:discriminator_output.size(0) // 2]
            real_predictions = discriminator_output[discriminator_output.size(0) // 2:]

        losses = {}
        losses['fake'] = self.gan_loss(fake_predictions, False, for_discriminator=True)
        losses['real'] = self.gan_loss(real_predictions, True, for_discriminator=True)

        total_loss = sum(losses.values()).mean()

        self.dis_optimizer.zero_grad()
        if self.config.fp16:
            from apex import amp
            with amp.scale_loss(total_loss, self.dis_optimizer, loss_id=1) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        self.dis_optimizer.step()

        return total_loss, losses


class Visualizer:
    def __init__(self, tensorboard_writer):
        self.writer = tensorboard_writer

    def log_training_images(self, step, data, generated, config):
        idx = 0
        grid = make_image_grid([
            (data['cloth'][0].cpu() / 2 + 0.5),
            (data['cloth_mask'][0].cpu()).expand(3, -1, -1),
            ((data['densepose'].cpu()[0] + 1) / 2),
            visualize_segmap(data['agnostic_parse'].cpu(), batch=idx),
            (data['warped_cloth'][idx].cpu() / 2 + 0.5),
            (data['person_rep'][idx].cpu() / 2 + 0.5),
            (data['densepose'][idx].cpu() / 2 + 0.5),
            visualize_segmap(data['segmentation'].cpu(), batch=idx) if data[
                                                                           'segmentation'] is not None else torch.zeros(
                3, 1024, 768),
            (generated[idx].cpu() / 2 + 0.5),
            (data['target'][idx].cpu() / 2 + 0.5)
        ], nrow=4)

        self.writer.add_images('train_images', grid.unsqueeze(0), step)

    def log_losses(self, step, gen_loss, gen_losses, dis_loss, dis_losses):
        self.writer.add_scalar('Loss/generator', gen_loss, step)
        self.writer.add_scalar('Loss/generator/adversarial', gen_losses['adversarial'].mean().item(), step)

        if 'feature_matching' in gen_losses:
            self.writer.add_scalar('Loss/generator/feature', gen_losses['feature_matching'].mean().item(), step)
        if 'perceptual' in gen_losses:
            self.writer.add_scalar('Loss/generator/perceptual', gen_losses['perceptual'].mean().item(), step)

        self.writer.add_scalar('Loss/discriminator', dis_loss, step)
        self.writer.add_scalar('Loss/discriminator/fake', dis_losses['fake'].mean().item(), step)
        self.writer.add_scalar('Loss/discriminator/real', dis_losses['real'].mean().item(), step)


def execute_training(config, train_loader, test_loader, test_vis_loader,
                     tensorboard_writer, warping_net, generator, discriminator, lpips_model):
    if not config.GT:
        warping_net.cuda()
        warping_net.eval()
    generator.train()
    discriminator.train()
    lpips_model.eval()

    trainer = GANTrainer(config, generator, discriminator, warping_net)
    visualizer = Visualizer(tensorboard_writer)

    for step in tqdm(range(config.load_step, config.keep_step + config.decay_step)):
        start_time = time.time()
        batch_data = train_loader.next_batch()

        prepared_data = trainer.prepare_inputs(batch_data, use_gt=config.GT, paired=True)

        gen_loss, gen_losses, generated_image = trainer.train_generator_step(prepared_data)
        dis_loss, dis_losses = trainer.train_discriminator_step(prepared_data)

        if (step + 1) % config.tensorboard_count == 0:
            visualizer.log_training_images(step + 1, prepared_data, generated_image, config)
            visualizer.log_losses(step + 1, gen_loss.item(), gen_losses, dis_loss.item(), dis_losses)

            generator.eval()
            test_batch = test_vis_loader.next_batch()
            test_prepared = trainer.prepare_inputs(test_batch, use_gt=config.GT, paired=False)

            with torch.no_grad():
                test_output = generator(
                    torch.cat((
                        test_prepared['person_rep'],
                        test_prepared['densepose'],
                        test_prepared['warped_cloth']
                    ), dim=1),
                    test_prepared['parse_map']
                )

                for i in range(config.num_test_visualize):
                    grid = make_image_grid([
                        (test_prepared['cloth'][i].cpu() / 2 + 0.5),
                        (test_prepared['cloth_mask'][i].cpu()).expand(3, -1, -1),
                        ((test_prepared['densepose'].cpu()[i] + 1) / 2),
                        visualize_segmap(test_prepared['agnostic_parse'].cpu(), batch=i),
                        (test_prepared['warped_cloth'][i].cpu() / 2 + 0.5),
                        (test_prepared['person_rep'][i].cpu() / 2 + 0.5),
                        (test_prepared['densepose'][i].cpu() / 2 + 0.5),
                        visualize_segmap(test_prepared['segmentation'].cpu(), batch=i) if test_prepared[
                                                                                              'segmentation'] is not None else torch.zeros(
                            3, 1024, 768),
                        (test_output[i].cpu() / 2 + 0.5),
                        (test_prepared['target'][i].cpu() / 2 + 0.5)
                    ], nrow=4)
                    tensorboard_writer.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)

            generator.train()

        if (step + 1) % config.lpips_count == 0:
            generator.eval()
            resize_transform = transforms.Compose([transforms.Resize((128, 128))])
            total_distance = 0.0

            with torch.no_grad():
                print("Evaluating LPIPS...")
                for i in tqdm(range(500)):
                    eval_batch = test_loader.next_batch()
                    eval_prepared = trainer.prepare_inputs(eval_batch, use_gt=config.GT, paired=True)

                    eval_output = generator(
                        torch.cat((
                            eval_prepared['person_rep'],
                            eval_prepared['densepose'],
                            eval_prepared['warped_cloth']
                        ), dim=1),
                        eval_prepared['parse_map']
                    )

                    total_distance += lpips_model.forward(
                        resize_transform(eval_prepared['target']),
                        resize_transform(eval_output)
                    )

            avg_distance = total_distance / 500
            print(f"LPIPS: {avg_distance}")
            tensorboard_writer.add_scalar('test/LPIPS', avg_distance, step + 1)
            generator.train()

        if (step + 1) % config.display_count == 0:
            elapsed = time.time() - start_time
            print(f"Step: {step + 1:8d}, Time: {elapsed:.3f}s, "
                  f"G_loss: {gen_loss.item():.4f}, G_adv: {gen_losses['adversarial'].mean().item():.4f}, "
                  f"D_loss: {dis_loss.item():.4f}, D_fake: {dis_losses['fake'].mean().item():.4f}, "
                  f"D_real: {dis_losses['real'].mean().item():.4f}", flush=True)

        if (step + 1) % config.save_count == 0:
            save_model_checkpoint(
                generator.module,
                os.path.join(config.checkpoint_dir, config.name, f'gen_step_{step + 1:06d}.pth'),
                config
            )
            save_model_checkpoint(
                discriminator.module,
                os.path.join(config.checkpoint_dir, config.name, f'dis_step_{step + 1:06d}.pth'),
                config
            )

        if (step + 1) % 1000 == 0:
            trainer.gen_scheduler.step()
            trainer.dis_scheduler.step()


def main():
    config_manager = TrainingConfig()
    config = config_manager.get()
    print(config)
    print(f"Starting training: {config.name}")

    train_dataset = VirtualTryOnDataset(config)
    train_loader = VirtualTryOnDataLoader(config, train_dataset)

    config.batch_size = 1
    config.dataroot = config.test_dataroot
    config.datamode = 'test'
    config.data_list = config.test_data_list
    test_dataset = VirtualTryOnDataset(config)
    test_dataset = Subset(test_dataset, np.arange(500))
    test_loader = VirtualTryOnDataLoader(config, test_dataset)

    config.batch_size = config.num_test_visualize
    test_vis_dataset = VirtualTryOnDataset(config)
    test_vis_loader = VirtualTryOnDataLoader(config, test_vis_dataset)

    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.name))

    warping_net = None
    if not config.GT:
        cloth_channels = 4
        context_channels = config.semantic_nc + 3
        output_channels = 13
        feature_channels = 96

        warping_net = WarpingFlowGenerator(
            config,
            input1_nc=cloth_channels,
            input2_nc=context_channels,
            output_nc=output_channels,
            ngf=feature_channels,
            norm_layer=nn.BatchNorm2d
        )
        load_model_checkpoint(warping_net, config.tocg_checkpoint)

    generator = SPADEGenerator(config, 3 + 3 + 3)
    generator.print_network()

    if len(config.gpu_ids) > 0:
        assert torch.cuda.is_available()
        generator.cuda()

    generator.init_weights(config.init_type, config.init_variance)
    discriminator = create_network(MultiscaleDiscriminator, config)

    lpips_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

    if config.gen_checkpoint and os.path.exists(config.gen_checkpoint):
        load_model_checkpoint(generator, config.gen_checkpoint)
        load_model_checkpoint(discriminator, config.dis_checkpoint)

    execute_training(
        config, train_loader, test_loader, test_vis_loader,
        tensorboard_writer, warping_net, generator, discriminator, lpips_model
    )

    save_model_checkpoint(
        generator,
        os.path.join(config.checkpoint_dir, config.name, 'gen_model_final.pth'),
        config
    )
    save_model_checkpoint(
        discriminator,
        os.path.join(config.checkpoint_dir, config.name, 'dis_model_final.pth'),
        config
    )

    print(f"Training completed: {config.name}")


if __name__ == "__main__":
    main()