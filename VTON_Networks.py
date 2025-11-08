
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from torch.nn.utils import spectral_norm
import numpy as np
import functools



class WarpingFlowGenerator(nn.Module):

    def __init__(self, config, cloth_channels, pose_channels, output_channels,
                 base_filters=64, norm_layer=nn.BatchNorm2d):
        super(WarpingFlowGenerator, self).__init__()
        self.warp_feature_type = config.warp_feature
        self.output_layer_type = config.out_layer

        # Cloth feature encoder (5 downsampling stages)
        self.cloth_encoder = nn.Sequential(
            ResidualBlock(cloth_channels, base_filters, norm_layer=norm_layer, scale='down'),  # 128
            ResidualBlock(base_filters, base_filters * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResidualBlock(base_filters * 2, base_filters * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down')  # 8
        )

        # Pose feature encoder (5 downsampling stages)
        self.pose_encoder = nn.Sequential(
            ResidualBlock(pose_channels, base_filters, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters, base_filters * 2, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters * 2, base_filters * 4, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down')
        )

        # Bottleneck processing
        self.bottleneck_conv = ResidualBlock(base_filters * 4, base_filters * 8,
                                             norm_layer=norm_layer, scale='same')

        # Segmentation decoder configuration based on warping strategy
        if config.warp_feature == 'T1':
            # Input channels: decoder_output + skip_connection + warped_features
            self.segmentation_decoder = nn.Sequential(
                ResidualBlock(base_filters * 8, base_filters * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResidualBlock(base_filters * 4 * 2 + base_filters * 4, base_filters * 4,
                              norm_layer=norm_layer, scale='up'),  # 32
                ResidualBlock(base_filters * 4 * 2 + base_filters * 4, base_filters * 2,
                              norm_layer=norm_layer, scale='up'),  # 64
                ResidualBlock(base_filters * 2 * 2 + base_filters * 4, base_filters,
                              norm_layer=norm_layer, scale='up'),  # 128
                ResidualBlock(base_filters * 1 * 2 + base_filters * 4, base_filters,
                              norm_layer=norm_layer, scale='up')  # 256
            )
        elif config.warp_feature == 'encoder':
            # Input channels: [decoder_output, skip_connection, warped_encoder_features]
            self.segmentation_decoder = nn.Sequential(
                ResidualBlock(base_filters * 8, base_filters * 4, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 4 * 3, base_filters * 4, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 4 * 3, base_filters * 2, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 2 * 3, base_filters, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 1 * 3, base_filters, norm_layer=norm_layer, scale='up')
            )

        # Output layer configuration
        if config.out_layer == 'relu':
            self.output_layer = ResidualBlock(
                base_filters + cloth_channels + pose_channels,
                output_channels,
                norm_layer=norm_layer,
                scale='same'
            )
        elif config.out_layer == 'conv':
            self.output_layer = nn.Sequential(
                ResidualBlock(base_filters + cloth_channels + pose_channels,
                              base_filters, norm_layer=norm_layer, scale='same'),
                nn.Conv2d(base_filters, output_channels, kernel_size=1, bias=True)
            )

        # 1x1 convolutions for cloth feature projection
        self.cloth_projection = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
        )

        # 1x1 convolutions for pose feature projection
        self.pose_projection = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
        )

        # Flow prediction layers at each pyramid level
        self.flow_predictors = nn.ModuleList([
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ])

        # Bottleneck refinement layers
        self.bottleneck_refinement = nn.Sequential(
            nn.Sequential(nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3,
                                    stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3,
                                    stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3,
                                    stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(base_filters, base_filters * 4, kernel_size=3,
                                    stride=1, padding=1, bias=True), nn.ReLU()),
        )

    def normalize_features(self, features):
        return features

    def forward(self, config, cloth_input, pose_input, upsample_mode='bilinear'):

        cloth_features = []
        pose_features = []
        flow_pyramid = []

        # Build feature pyramid through encoders
        for pyramid_level in range(5):
            if pyramid_level == 0:
                cloth_features.append(self.cloth_encoder[pyramid_level](cloth_input))
                pose_features.append(self.pose_encoder[pyramid_level](pose_input))
            else:
                cloth_features.append(
                    self.cloth_encoder[pyramid_level](cloth_features[pyramid_level - 1])
                )
                pose_features.append(
                    self.pose_encoder[pyramid_level](pose_features[pyramid_level - 1])
                )

        # Compute optical flow at each pyramid level (coarse-to-fine)
        for pyramid_level in range(5):
            # Process from finest to coarsest features
            level_idx = 4 - pyramid_level
            batch_size, _, height, width = cloth_features[level_idx].size()
            sampling_grid = create_sampling_grid(batch_size, height, width, config)

            if pyramid_level == 0:
                # Coarsest level - initialize flow
                cloth_descriptor = cloth_features[level_idx]  # (base_filters * 4) x 8 x 6
                pose_descriptor = pose_features[level_idx]
                combined_features = torch.cat([cloth_descriptor, pose_descriptor], dim=1)

                # Predict initial flow
                flow = self.flow_predictors[pyramid_level](
                    self.normalize_features(combined_features)
                ).permute(0, 2, 3, 1)
                flow_pyramid.append(flow)

                # Initialize decoder
                decoder_features = self.bottleneck_conv(pose_descriptor)
                decoder_features = self.segmentation_decoder[pyramid_level](decoder_features)

            else:
                # Finer levels - refine flow iteratively
                # Upsample and project features
                cloth_descriptor = F.interpolate(
                    cloth_descriptor, scale_factor=2, mode=upsample_mode
                ) + self.cloth_projection[level_idx](cloth_features[level_idx])

                pose_descriptor = F.interpolate(
                    pose_descriptor, scale_factor=2, mode=upsample_mode
                ) + self.pose_projection[level_idx](pose_features[level_idx])

                # Upsample previous flow
                upsampled_flow = F.interpolate(
                    flow_pyramid[pyramid_level - 1].permute(0, 3, 1, 2),
                    scale_factor=2,
                    mode=upsample_mode
                ).permute(0, 2, 3, 1)

                # Normalize flow for grid sampling
                flow_normalized = torch.cat([
                    upsampled_flow[:, :, :, 0:1] / ((width / 2 - 1.0) / 2.0),
                    upsampled_flow[:, :, :, 1:2] / ((height / 2 - 1.0) / 2.0)
                ], dim=3)

                # Warp cloth features using current flow
                warped_cloth_descriptor = F.grid_sample(
                    cloth_descriptor,
                    flow_normalized + sampling_grid,
                    padding_mode='border'
                )

                # Refine flow based on warped features
                flow_refinement_input = torch.cat([
                    warped_cloth_descriptor,
                    self.bottleneck_refinement[pyramid_level - 1](decoder_features)
                ], dim=1)

                flow_delta = self.flow_predictors[pyramid_level](
                    self.normalize_features(flow_refinement_input)
                ).permute(0, 2, 3, 1)

                flow = upsampled_flow + flow_delta
                flow_pyramid.append(flow)

                # Update decoder with warped features
                if self.warp_feature_type == 'T1':
                    decoder_features = self.segmentation_decoder[pyramid_level](
                        torch.cat([decoder_features, pose_features[level_idx],
                                   warped_cloth_descriptor], dim=1)
                    )
                elif self.warp_feature_type == 'encoder':
                    warped_encoder_features = F.grid_sample(
                        cloth_features[level_idx],
                        flow_normalized + sampling_grid,
                        padding_mode='border'
                    )
                    decoder_features = self.segmentation_decoder[pyramid_level](
                        torch.cat([decoder_features, pose_features[level_idx],
                                   warped_encoder_features], dim=1)
                    )

        # Final upsampling and warping at original resolution
        batch_size, _, height, width = cloth_input.size()
        sampling_grid = create_sampling_grid(batch_size, height, width, config)

        final_flow = F.interpolate(
            flow_pyramid[-1].permute(0, 3, 1, 2),
            scale_factor=2,
            mode=upsample_mode
        ).permute(0, 2, 3, 1)

        final_flow_normalized = torch.cat([
            final_flow[:, :, :, 0:1] / ((width / 2 - 1.0) / 2.0),
            final_flow[:, :, :, 1:2] / ((height / 2 - 1.0) / 2.0)
        ], dim=3)

        warped_input = F.grid_sample(
            cloth_input,
            final_flow_normalized + sampling_grid,
            padding_mode='border'
        )

        # Generate final output
        segmentation_output = self.output_layer(
            torch.cat([decoder_features, pose_input, warped_input], dim=1)
        )

        # Split warped cloth and mask
        warped_cloth = warped_input[:, :-1, :, :]
        warped_cloth_mask = warped_input[:, -1:, :, :]

        return flow_pyramid, segmentation_output, warped_cloth, warped_cloth_mask


def create_sampling_grid(batch_size, height, width, config):

    grid_x = torch.linspace(-1.0, 1.0, width).view(1, 1, width, 1).expand(
        batch_size, height, -1, -1
    )
    grid_y = torch.linspace(-1.0, 1.0, height).view(1, height, 1, 1).expand(
        batch_size, -1, width, -1
    )

    grid = torch.cat([grid_x, grid_y], dim=3)

    if config.cuda:
        grid = grid.cuda()

    return grid


class ResidualBlock(nn.Module):


    def __init__(self, in_channels, out_channels, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)

        assert scale in ['up', 'down', 'same'], \
            "ResidualBlock scale must be 'up', 'down', or 'same'"

        # Scale transformation for skip connection
        if scale == 'same':
            self.scale_transform = nn.Conv2d(in_channels, out_channels,
                                             kernel_size=1, bias=True)
        elif scale == 'up':
            self.scale_transform = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            )
        elif scale == 'down':
            self.scale_transform = nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, stride=2, padding=1,
                                             bias=use_bias)

        # Main residual path
        self.residual_path = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=use_bias),
            norm_layer(out_channels)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass with residual connection"""
        skip_connection = self.scale_transform(x)
        residual = self.residual_path(skip_connection)
        return self.activation(skip_connection + residual)


class VGG19FeatureExtractor(nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG19FeatureExtractor, self).__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features

        # Split VGG into 5 feature extraction stages
        self.stage_1 = nn.Sequential()  # Up to relu1_2
        self.stage_2 = nn.Sequential()  # Up to relu2_2
        self.stage_3 = nn.Sequential()  # Up to relu3_4
        self.stage_4 = nn.Sequential()  # Up to relu4_4
        self.stage_5 = nn.Sequential()  # Up to relu5_4

        for x in range(2):
            self.stage_1.add_module(str(x), vgg_pretrained[x])
        for x in range(2, 7):
            self.stage_2.add_module(str(x), vgg_pretrained[x])
        for x in range(7, 12):
            self.stage_3.add_module(str(x), vgg_pretrained[x])
        for x in range(12, 21):
            self.stage_4.add_module(str(x), vgg_pretrained[x])
        for x in range(21, 30):
            self.stage_5.add_module(str(x), vgg_pretrained[x])

        # Freeze parameters if not training
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_tensor):
        """Extract multi-scale features"""
        features_1 = self.stage_1(input_tensor)
        features_2 = self.stage_2(features_1)
        features_3 = self.stage_3(features_2)
        features_4 = self.stage_4(features_3)
        features_5 = self.stage_5(features_4)

        return [features_1, features_2, features_3, features_4, features_5]


class PerceptualLoss(nn.Module):


    def __init__(self, config, layer_ids=None):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGG19FeatureExtractor()

        if config.cuda:
            self.feature_extractor.cuda()

        self.criterion = nn.L1Loss()
        self.layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layer_ids = layer_ids

    def forward(self, prediction, target):
        """Compute perceptual loss between prediction and target"""
        pred_features = self.feature_extractor(prediction)
        target_features = self.feature_extractor(target)

        total_loss = 0

        # Use all layers if not specified
        if self.layer_ids is None:
            self.layer_ids = list(range(len(pred_features)))

        # Weighted sum of feature distances
        for layer_id in self.layer_ids:
            layer_loss = self.criterion(
                pred_features[layer_id],
                target_features[layer_id].detach()
            )
            total_loss += self.layer_weights[layer_id] * layer_loss

        return total_loss


class AdversarialLossFunction(nn.Module):

    def __init__(self, use_least_squares=True, real_label=1.0, fake_label=0.0,
                 tensor_type=torch.FloatTensor):
        super(AdversarialLossFunction, self).__init__()
        self.real_label_value = real_label
        self.fake_label_value = fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.tensor_type = tensor_type

        if use_least_squares:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.BCELoss()

    def create_target_tensor(self, input_tensor, is_real_target):
        """Create target label tensor matching input shape"""
        if is_real_target:
            if (self.real_label_tensor is None or
                    self.real_label_tensor.numel() != input_tensor.numel()):
                self.real_label_tensor = self.tensor_type(input_tensor.size()).fill_(
                    self.real_label_value
                )
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor
        else:
            if (self.fake_label_tensor is None or
                    self.fake_label_tensor.numel() != input_tensor.numel()):
                self.fake_label_tensor = self.tensor_type(input_tensor.size()).fill_(
                    self.fake_label_value
                )
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor

    def __call__(self, discriminator_output, is_real_target):
        # Handle multiscale discriminator with nested outputs
        if isinstance(discriminator_output[0], list):
            total_loss = 0
            for scale_output in discriminator_output:
                prediction = scale_output[-1]
                target = self.create_target_tensor(prediction, is_real_target)
                total_loss += self.loss_fn(prediction, target)
            return total_loss
        else:
            # Single scale discriminator
            prediction = discriminator_output[-1]
            target = self.create_target_tensor(prediction, is_real_target)
            return self.loss_fn(prediction, target)


class MultiScalePatchDiscriminator(nn.Module):

    def __init__(self, input_channels, base_filters=64, num_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_scales=3,
                 extract_features=False, downsample_input=False,
                 use_dropout=False, use_spectral_norm=False):
        super(MultiScalePatchDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.num_layers = num_layers
        self.extract_features = extract_features
        self.downsample_input = downsample_input

        # Create discriminator for each scale
        for scale_idx in range(num_scales):
            discriminator = PatchDiscriminator(
                input_channels, base_filters, num_layers, norm_layer,
                use_sigmoid, extract_features, use_dropout, use_spectral_norm
            )

            if extract_features:
                # Register individual layers for feature extraction
                for layer_idx in range(num_layers + 2):
                    setattr(
                        self,
                        f'scale_{scale_idx}_layer_{layer_idx}',
                        getattr(discriminator, f'model_{layer_idx}')
                    )
            else:
                setattr(self, f'discriminator_{scale_idx}', discriminator.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def forward_single_scale(self, layers, input_tensor):
        """Forward pass for single scale"""
        if self.extract_features:
            features = [input_tensor]
            for layer in layers:
                features.append(layer(features[-1]))
            return features[1:]
        else:
            return [layers(input_tensor)]

    def forward(self, input_tensor):
        """Forward pass through all scales"""
        all_outputs = []

        # Downsample input if configured
        current_input = self.downsample(input_tensor) if self.downsample_input else input_tensor

        for scale_idx in range(self.num_scales):
            if self.extract_features:
                layers = [
                    getattr(self, f'scale_{self.num_scales - 1 - scale_idx}_layer_{j}')
                    for j in range(self.num_layers + 2)
                ]
            else:
                layers = getattr(self, f'discriminator_{self.num_scales - 1 - scale_idx}')

            all_outputs.append(self.forward_single_scale(layers, current_input))

            # Downsample for next scale (except last)
            if scale_idx != (self.num_scales - 1):
                current_input = self.downsample(current_input)

        return all_outputs


class PatchDiscriminator(nn.Module):


    def __init__(self, input_channels, base_filters=64, num_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 extract_features=False, use_dropout=False, use_spectral_norm=False):
        super(PatchDiscriminator, self).__init__()
        self.extract_features = extract_features
        self.num_layers = num_layers
        self.spectral_norm_fn = spectral_norm if use_spectral_norm else lambda x: x

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1.0) / 2))

        # Build layer sequence
        layer_sequence = [[
            nn.Conv2d(input_channels, base_filters, kernel_size=kernel_size,
                      stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        ]]

        num_filters = base_filters
        for layer_idx in range(1, num_layers):
            prev_filters = num_filters
            num_filters = min(num_filters * 2, 512)

            layer_block = [
                self.spectral_norm_fn(
                    nn.Conv2d(prev_filters, num_filters, kernel_size=kernel_size,
                              stride=2, padding=padding)
                ),
                norm_layer(num_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ]

            if use_dropout:
                layer_block.append(nn.Dropout(0.5))

            layer_sequence.append(layer_block)

        # Penultimate layer
        prev_filters = num_filters
        num_filters = min(num_filters * 2, 512)
        layer_sequence.append([
            nn.Conv2d(prev_filters, num_filters, kernel_size=kernel_size,
                      stride=1, padding=padding),
            norm_layer(num_filters),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Output layer
        layer_sequence.append([
            nn.Conv2d(num_filters, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ])

        if use_sigmoid:
            layer_sequence.append([nn.Sigmoid()])

        # Create model architecture
        if extract_features:
            for idx, layers in enumerate(layer_sequence):
                setattr(self, f'model_{idx}', nn.Sequential(*layers))
        else:
            all_layers = []
            for layers in layer_sequence:
                all_layers.extend(layers)
            self.model = nn.Sequential(*all_layers)

    def forward(self, input_tensor):
        """Forward pass"""
        if self.extract_features:
            features = [input_tensor]
            for layer_idx in range(self.num_layers + 2):
                layer = getattr(self, f'model_{layer_idx}')
                features.append(layer(features[-1]))
            return features[1:]
        else:
            return self.model(input_tensor)


# Utility functions

def save_model_checkpoint(model, save_path, config):
    """Save model checkpoint to disk"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu().state_dict(), save_path)

    if config.cuda:
        model.cuda()


def load_model_checkpoint(model, checkpoint_path, config):
    """Load model checkpoint from disk"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(f"Model loaded from {checkpoint_path}")

    if config.cuda:
        model.cuda()

    return load_log


def initialize_network_weights(module):
    class_name = module.__class__.__name__

    if 'Conv2d' in class_name:
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif 'BatchNorm2d' in class_name:
        module.weight.data.normal_(mean=1.0, std=0.02)
        module.bias.data.fill_(0)


def get_normalization_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            f"Normalization type '{norm_type}' not implemented. "
            f"Choose 'batch' or 'instance'"
        )


def create_discriminator(input_channels, base_filters=64, num_layers=3,
                         norm_type='instance', use_sigmoid=False, num_scales=2,
                         extract_features=False, gpu_ids=None, downsample_input=False,
                         use_dropout=False, use_spectral_norm=False):

    if gpu_ids is None:
        gpu_ids = []

    norm_layer = get_normalization_layer(norm_type=norm_type)

    discriminator = MultiScalePatchDiscriminator(
        input_channels=input_channels,
        base_filters=base_filters,
        num_layers=num_layers,
        norm_layer=norm_layer,
        use_sigmoid=use_sigmoid,
        num_scales=num_scales,
        extract_features=extract_features,
        downsample_input=downsample_input,
        use_dropout=use_dropout,
        use_spectral_norm=use_spectral_norm
    )

    print(discriminator)

    if len(gpu_ids) > 0:
        assert torch.cuda.is_available(), "CUDA not available but GPU IDs provided"
        discriminator.cuda()

    discriminator.apply(initialize_network_weights)

    return discriminator

class WarpingFlowGenerator(nn.Module):
    """
    Generates optical flow to warp clothing onto target pose
    Uses Feature Pyramid Network with iterative flow refinement
    """

    def __init__(self, config, cloth_channels, pose_channels, output_channels,
                 base_filters=64, norm_layer=nn.BatchNorm2d):
        super(WarpingFlowGenerator, self).__init__()
        self.warp_feature_type = config.warp_feature
        self.output_layer_type = config.out_layer

        # Cloth feature encoder (5 downsampling stages)
        self.cloth_encoder = nn.Sequential(
            ResidualBlock(cloth_channels, base_filters, norm_layer=norm_layer, scale='down'),      # 128
            ResidualBlock(base_filters, base_filters * 2, norm_layer=norm_layer, scale='down'),    # 64
            ResidualBlock(base_filters * 2, base_filters * 4, norm_layer=norm_layer, scale='down'), # 32
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down'), # 16
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down')  # 8
        )

        # Pose feature encoder (5 downsampling stages)
        self.pose_encoder = nn.Sequential(
            ResidualBlock(pose_channels, base_filters, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters, base_filters * 2, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters * 2, base_filters * 4, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down'),
            ResidualBlock(base_filters * 4, base_filters * 4, norm_layer=norm_layer, scale='down')
        )

        # Bottleneck processing
        self.bottleneck_conv = ResidualBlock(base_filters * 4, base_filters * 8,
                                             norm_layer=norm_layer, scale='same')

        # Segmentation decoder configuration based on warping strategy
        if config.warp_feature == 'T1':
            # Input channels: decoder_output + skip_connection + warped_features
            self.segmentation_decoder = nn.Sequential(
                ResidualBlock(base_filters * 8, base_filters * 4, norm_layer=norm_layer, scale='up'),           # 16
                ResidualBlock(base_filters * 4 * 2 + base_filters * 4, base_filters * 4,
                            norm_layer=norm_layer, scale='up'),  # 32
                ResidualBlock(base_filters * 4 * 2 + base_filters * 4, base_filters * 2,
                            norm_layer=norm_layer, scale='up'),  # 64
                ResidualBlock(base_filters * 2 * 2 + base_filters * 4, base_filters,
                            norm_layer=norm_layer, scale='up'),  # 128
                ResidualBlock(base_filters * 1 * 2 + base_filters * 4, base_filters,
                            norm_layer=norm_layer, scale='up')   # 256
            )
        elif config.warp_feature == 'encoder':
            # Input channels: [decoder_output, skip_connection, warped_encoder_features]
            self.segmentation_decoder = nn.Sequential(
                ResidualBlock(base_filters * 8, base_filters * 4, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 4 * 3, base_filters * 4, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 4 * 3, base_filters * 2, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 2 * 3, base_filters, norm_layer=norm_layer, scale='up'),
                ResidualBlock(base_filters * 1 * 3, base_filters, norm_layer=norm_layer, scale='up')
            )

        # Output layer configuration
        if config.out_layer == 'relu':
            self.output_layer = ResidualBlock(
                base_filters + cloth_channels + pose_channels,
                output_channels,
                norm_layer=norm_layer,
                scale='same'
            )
        elif config.out_layer == 'conv':
            self.output_layer = nn.Sequential(
                ResidualBlock(base_filters + cloth_channels + pose_channels,
                            base_filters, norm_layer=norm_layer, scale='same'),
                nn.Conv2d(base_filters, output_channels, kernel_size=1, bias=True)
            )

        # 1x1 convolutions for cloth feature projection
        self.cloth_projection = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
        )

        # 1x1 convolutions for pose feature projection
        self.pose_projection = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=1, bias=True),
        )

        # Flow prediction layers at each pyramid level
        self.flow_predictors = nn.ModuleList([
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(base_filters * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ])

        # Bottleneck refinement layers
        self.bottleneck_refinement = nn.Sequential(
            nn.Sequential(nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3,
                                   stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3,
                                   stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3,
                                   stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(base_filters, base_filters * 4, kernel_size=3,
                                   stride=1, padding=1, bias=True), nn.ReLU()),
        )

    def normalize_features(self, features):
     return features

    def forward(self, config, cloth_input, pose_input, upsample_mode='bilinear'):

        cloth_features = []
        pose_features = []
        flow_pyramid = []

        # Build feature pyramid through encoders
        for pyramid_level in range(5):
            if pyramid_level == 0:
                cloth_features.append(self.cloth_encoder[pyramid_level](cloth_input))
                pose_features.append(self.pose_encoder[pyramid_level](pose_input))
            else:
                cloth_features.append(
                    self.cloth_encoder[pyramid_level](cloth_features[pyramid_level - 1])
                )
                pose_features.append(
                    self.pose_encoder[pyramid_level](pose_features[pyramid_level - 1])
                )

        # Compute optical flow at each pyramid level (coarse-to-fine)
        for pyramid_level in range(5):
            # Process from finest to coarsest features
            level_idx = 4 - pyramid_level
            batch_size, _, height, width = cloth_features[level_idx].size()
            sampling_grid = create_sampling_grid(batch_size, height, width, config)

            if pyramid_level == 0:
                # Coarsest level - initialize flow
                cloth_descriptor = cloth_features[level_idx]  # (base_filters * 4) x 8 x 6
                pose_descriptor = pose_features[level_idx]
                combined_features = torch.cat([cloth_descriptor, pose_descriptor], dim=1)

                # Predict initial flow
                flow = self.flow_predictors[pyramid_level](
                    self.normalize_features(combined_features)
                ).permute(0, 2, 3, 1)
                flow_pyramid.append(flow)

                # Initialize decoder
                decoder_features = self.bottleneck_conv(pose_descriptor)
                decoder_features = self.segmentation_decoder[pyramid_level](decoder_features)

            else:
                # Finer levels - refine flow iteratively
                # Upsample and project features
                cloth_descriptor = F.interpolate(
                    cloth_descriptor, scale_factor=2, mode=upsample_mode
                ) + self.cloth_projection[level_idx](cloth_features[level_idx])

                pose_descriptor = F.interpolate(
                    pose_descriptor, scale_factor=2, mode=upsample_mode
                ) + self.pose_projection[level_idx](pose_features[level_idx])

                # Upsample previous flow
                upsampled_flow = F.interpolate(
                    flow_pyramid[pyramid_level - 1].permute(0, 3, 1, 2),
                    scale_factor=2,
                    mode=upsample_mode
                ).permute(0, 2, 3, 1)

                # Normalize flow for grid sampling
                flow_normalized = torch.cat([
                    upsampled_flow[:, :, :, 0:1] / ((width / 2 - 1.0) / 2.0),
                    upsampled_flow[:, :, :, 1:2] / ((height / 2 - 1.0) / 2.0)
                ], dim=3)

                # Warp cloth features using current flow
                warped_cloth_descriptor = F.grid_sample(
                    cloth_descriptor,
                    flow_normalized + sampling_grid,
                    padding_mode='border'
                )

                # Refine flow based on warped features
                flow_refinement_input = torch.cat([
                    warped_cloth_descriptor,
                    self.bottleneck_refinement[pyramid_level - 1](decoder_features)
                ], dim=1)

                flow_delta = self.flow_predictors[pyramid_level](
                    self.normalize_features(flow_refinement_input)
                ).permute(0, 2, 3, 1)

                flow = upsampled_flow + flow_delta
                flow_pyramid.append(flow)

                # Update decoder with warped features
                if self.warp_feature_type == 'T1':
                    decoder_features = self.segmentation_decoder[pyramid_level](
                        torch.cat([decoder_features, pose_features[level_idx],
                                 warped_cloth_descriptor], dim=1)
                    )
                elif self.warp_feature_type == 'encoder':
                    warped_encoder_features = F.grid_sample(
                        cloth_features[level_idx],
                        flow_normalized + sampling_grid,
                        padding_mode='border'
                    )
                    decoder_features = self.segmentation_decoder[pyramid_level](
                        torch.cat([decoder_features, pose_features[level_idx],
                                 warped_encoder_features], dim=1)
                    )

        # Final upsampling and warping at original resolution
        batch_size, _, height, width = cloth_input.size()
        sampling_grid = create_sampling_grid(batch_size, height, width, config)

        final_flow = F.interpolate(
            flow_pyramid[-1].permute(0, 3, 1, 2),
            scale_factor=2,
            mode=upsample_mode
        ).permute(0, 2, 3, 1)

        final_flow_normalized = torch.cat([
            final_flow[:, :, :, 0:1] / ((width / 2 - 1.0) / 2.0),
            final_flow[:, :, :, 1:2] / ((height / 2 - 1.0) / 2.0)
        ], dim=3)

        warped_input = F.grid_sample(
            cloth_input,
            final_flow_normalized + sampling_grid,
            padding_mode='border'
        )

        # Generate final output
        segmentation_output = self.output_layer(
            torch.cat([decoder_features, pose_input, warped_input], dim=1)
        )

        # Split warped cloth and mask
        warped_cloth = warped_input[:, :-1, :, :]
        warped_cloth_mask = warped_input[:, -1:, :, :]

        return flow_pyramid, segmentation_output, warped_cloth, warped_cloth_mask


def create_sampling_grid(batch_size, height, width, config):
    grid_x = torch.linspace(-1.0, 1.0, width).view(1, 1, width, 1).expand(
        batch_size, height, -1, -1
    )
    grid_y = torch.linspace(-1.0, 1.0, height).view(1, height, 1, 1).expand(
        batch_size, -1, width, -1
    )

    grid = torch.cat([grid_x, grid_y], dim=3)

    if config.cuda:
        grid = grid.cuda()

    return grid


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)

        assert scale in ['up', 'down', 'same'], \
            "ResidualBlock scale must be 'up', 'down', or 'same'"

        # Scale transformation for skip connection
        if scale == 'same':
            self.scale_transform = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, bias=True)
        elif scale == 'up':
            self.scale_transform = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            )
        elif scale == 'down':
            self.scale_transform = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=3, stride=2, padding=1,
                                            bias=use_bias)

        # Main residual path
        self.residual_path = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=use_bias),
            norm_layer(out_channels)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        skip_connection = self.scale_transform(x)
        residual = self.residual_path(skip_connection)
        return self.activation(skip_connection + residual)


class VGG19FeatureExtractor(nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG19FeatureExtractor, self).__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features

        # Split VGG into 5 feature extraction stages
        self.stage_1 = nn.Sequential()  # Up to relu1_2
        self.stage_2 = nn.Sequential()  # Up to relu2_2
        self.stage_3 = nn.Sequential()  # Up to relu3_4
        self.stage_4 = nn.Sequential()  # Up to relu4_4
        self.stage_5 = nn.Sequential()  # Up to relu5_4

        for x in range(2):
            self.stage_1.add_module(str(x), vgg_pretrained[x])
        for x in range(2, 7):
            self.stage_2.add_module(str(x), vgg_pretrained[x])
        for x in range(7, 12):
            self.stage_3.add_module(str(x), vgg_pretrained[x])
        for x in range(12, 21):
            self.stage_4.add_module(str(x), vgg_pretrained[x])
        for x in range(21, 30):
            self.stage_5.add_module(str(x), vgg_pretrained[x])

        # Freeze parameters if not training
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_tensor):
        features_1 = self.stage_1(input_tensor)
        features_2 = self.stage_2(features_1)
        features_3 = self.stage_3(features_2)
        features_4 = self.stage_4(features_3)
        features_5 = self.stage_5(features_4)

        return [features_1, features_2, features_3, features_4, features_5]


class PerceptualLoss(nn.Module):

    def __init__(self, config, layer_ids=None):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGG19FeatureExtractor()

        if config.cuda:
            self.feature_extractor.cuda()

        self.criterion = nn.L1Loss()
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layer_ids = layer_ids

    def forward(self, prediction, target):
        pred_features = self.feature_extractor(prediction)
        target_features = self.feature_extractor(target)

        total_loss = 0

        # Use all layers if not specified
        if self.layer_ids is None:
            self.layer_ids = list(range(len(pred_features)))

        # Weighted sum of feature distances
        for layer_id in self.layer_ids:
            layer_loss = self.criterion(
                pred_features[layer_id],
                target_features[layer_id].detach()
            )
            total_loss += self.layer_weights[layer_id] * layer_loss

        return total_loss


class AdversarialLossFunction(nn.Module):

    def __init__(self, use_least_squares=True, real_label=1.0, fake_label=0.0,
                 tensor_type=torch.FloatTensor):
        super(AdversarialLossFunction, self).__init__()
        self.real_label_value = real_label
        self.fake_label_value = fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.tensor_type = tensor_type

        if use_least_squares:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.BCELoss()

    def create_target_tensor(self, input_tensor, is_real_target):
        if is_real_target:
            if (self.real_label_tensor is None or
                self.real_label_tensor.numel() != input_tensor.numel()):
                self.real_label_tensor = self.tensor_type(input_tensor.size()).fill_(
                    self.real_label_value
                )
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor
        else:
            if (self.fake_label_tensor is None or
                self.fake_label_tensor.numel() != input_tensor.numel()):
                self.fake_label_tensor = self.tensor_type(input_tensor.size()).fill_(
                    self.fake_label_value
                )
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor

    def __call__(self, discriminator_output, is_real_target):
        # Handle multiscale discriminator with nested outputs
        if isinstance(discriminator_output[0], list):
            total_loss = 0
            for scale_output in discriminator_output:
                prediction = scale_output[-1]
                target = self.create_target_tensor(prediction, is_real_target)
                total_loss += self.loss_fn(prediction, target)
            return total_loss
        else:
            # Single scale discriminator
            prediction = discriminator_output[-1]
            target = self.create_target_tensor(prediction, is_real_target)
            return self.loss_fn(prediction, target)


class MultiScalePatchDiscriminator(nn.Module):

    def __init__(self, input_channels, base_filters=64, num_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_scales=3,
                 extract_features=False, downsample_input=False,
                 use_dropout=False, use_spectral_norm=False):
        super(MultiScalePatchDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.num_layers = num_layers
        self.extract_features = extract_features
        self.downsample_input = downsample_input

        # Create discriminator for each scale
        for scale_idx in range(num_scales):
            discriminator = PatchDiscriminator(
                input_channels, base_filters, num_layers, norm_layer,
                use_sigmoid, extract_features, use_dropout, use_spectral_norm
            )

            if extract_features:
                # Register individual layers for feature extraction
                for layer_idx in range(num_layers + 2):
                    setattr(
                        self,
                        f'scale_{scale_idx}_layer_{layer_idx}',
                        getattr(discriminator, f'model_{layer_idx}')
                    )
            else:
                setattr(self, f'discriminator_{scale_idx}', discriminator.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def forward_single_scale(self, layers, input_tensor):
        if self.extract_features:
            features = [input_tensor]
            for layer in layers:
                features.append(layer(features[-1]))
            return features[1:]
        else:
            return [layers(input_tensor)]

    def forward(self, input_tensor):
        all_outputs = []

        # Downsample input if configured
        current_input = self.downsample(input_tensor) if self.downsample_input else input_tensor

        for scale_idx in range(self.num_scales):
            if self.extract_features:
                layers = [
                    getattr(self, f'scale_{self.num_scales - 1 - scale_idx}_layer_{j}')
                    for j in range(self.num_layers + 2)
                ]
            else:
                layers = getattr(self, f'discriminator_{self.num_scales - 1 - scale_idx}')

            all_outputs.append(self.forward_single_scale(layers, current_input))

            # Downsample for next scale (except last)
            if scale_idx != (self.num_scales - 1):
                current_input = self.downsample(current_input)

        return all_outputs


class PatchDiscriminator(nn.Module):

    def __init__(self, input_channels, base_filters=64, num_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 extract_features=False, use_dropout=False, use_spectral_norm=False):
        super(PatchDiscriminator, self).__init__()
        self.extract_features = extract_features
        self.num_layers = num_layers
        self.spectral_norm_fn = spectral_norm if use_spectral_norm else lambda x: x

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1.0) / 2))

        # Build layer sequence
        layer_sequence = [[
            nn.Conv2d(input_channels, base_filters, kernel_size=kernel_size,
                     stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        ]]

        num_filters = base_filters
        for layer_idx in range(1, num_layers):
            prev_filters = num_filters
            num_filters = min(num_filters * 2, 512)

            layer_block = [
                self.spectral_norm_fn(
                    nn.Conv2d(prev_filters, num_filters, kernel_size=kernel_size,
                             stride=2, padding=padding)
                ),
                norm_layer(num_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ]

            if use_dropout:
                layer_block.append(nn.Dropout(0.5))

            layer_sequence.append(layer_block)

        # Penultimate layer
        prev_filters = num_filters
        num_filters = min(num_filters * 2, 512)
        layer_sequence.append([
            nn.Conv2d(prev_filters, num_filters, kernel_size=kernel_size,
                     stride=1, padding=padding),
            norm_layer(num_filters),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Output layer
        layer_sequence.append([
            nn.Conv2d(num_filters, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ])

        if use_sigmoid:
            layer_sequence.append([nn.Sigmoid()])

        # Create model architecture
        if extract_features:
            for idx, layers in enumerate(layer_sequence):
                setattr(self, f'model_{idx}', nn.Sequential(*layers))
        else:
            all_layers = []
            for layers in layer_sequence:
                all_layers.extend(layers)
            self.model = nn.Sequential(*all_layers)

    def forward(self, input_tensor):
        if self.extract_features:
            features = [input_tensor]
            for layer_idx in range(self.num_layers + 2):
                layer = getattr(self, f'model_{layer_idx}')
                features.append(layer(features[-1]))
            return features[1:]
        else:
            return self.model(input_tensor)


# Utility functions

def save_model_checkpoint(model, save_path, config):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu().state_dict(), save_path)

    if config.cuda:
        model.cuda()


def load_model_checkpoint(model, checkpoint_path, config):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(f"Model loaded from {checkpoint_path}")

    if config.cuda:
        model.cuda()

    return load_log


def initialize_network_weights(module):
    class_name = module.__class__.__name__

    if 'Conv2d' in class_name:
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif 'BatchNorm2d' in class_name:
        module.weight.data.normal_(mean=1.0, std=0.02)
        module.bias.data.fill_(0)


def get_normalization_layer(norm_type='instance'):

    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            f"Normalization type '{norm_type}' not implemented. "
            f"Choose 'batch' or 'instance'"
        )


def create_discriminator(input_channels, base_filters=64, num_layers=3,
                        norm_type='instance', use_sigmoid=False, num_scales=2,
                        extract_features=False, gpu_ids=None, downsample_input=False,
                        use_dropout=False, use_spectral_norm=False):

    if gpu_ids is None:
        gpu_ids = []

    norm_layer = get_normalization_layer(norm_type=norm_type)

    discriminator = MultiScalePatchDiscriminator(
        input_channels=input_channels,
        base_filters=base_filters,
        num_layers=num_layers,
        norm_layer=norm_layer,
        use_sigmoid=use_sigmoid,
        num_scales=num_scales,
        extract_features=extract_features,
        downsample_input=downsample_input,
        use_dropout=use_dropout,
        use_spectral_norm=use_spectral_norm
    )

    print(discriminator)

    if len(gpu_ids) > 0:
        assert torch.cuda.is_available(), "CUDA not available but GPU IDs provided"
        discriminator.cuda()

    discriminator.apply(initialize_network_weights)

    return discriminator