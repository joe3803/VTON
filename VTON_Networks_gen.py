import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm
import numpy as np


class NetworkBase(nn.Module):

    def __init__(self):
        super(NetworkBase, self).__init__()

    def print_network_info(self):

        total_params = sum(param.numel() for param in self.parameters())
        print(f"Network [{self.__class__.__name__}] created. "
              f"Total parameters: {total_params / 1e6:.1f}M. "
              f"Use print(network) to see full architecture.")

    def initialize_weights(self, init_method='normal', gain=0.02):


        def init_func(module):
            class_name = module.__class__.__name__

            # Initialize BatchNorm layers
            if 'BatchNorm2d' in class_name:
                if hasattr(module, 'weight') and module.weight is not None:
                    init.normal_(module.weight.data, mean=1.0, std=gain)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.constant_(module.bias.data, 0.0)

            # Initialize Conv and Linear layers
            elif ('Conv' in class_name or 'Linear' in class_name) and hasattr(module, 'weight'):
                if init_method == 'normal':
                    init.normal_(module.weight.data, mean=0.0, std=gain)
                elif init_method == 'xavier':
                    init.xavier_normal_(module.weight.data, gain=gain)
                elif init_method == 'xavier_uniform':
                    init.xavier_uniform_(module.weight.data, gain=1.0)
                elif init_method == 'kaiming':
                    init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
                elif init_method == 'orthogonal':
                    init.orthogonal_(module.weight.data, gain=gain)
                elif init_method == 'none':
                    module.reset_parameters()
                else:
                    raise NotImplementedError(
                        f"Initialization method '{init_method}' not implemented"
                    )

                if hasattr(module, 'bias') and module.bias is not None:
                    init.constant_(module.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, *inputs):

        raise NotImplementedError


class RegionMaskNormalization(nn.Module):

    def __init__(self, num_channels):
        super(RegionMaskNormalization, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=False)

    def normalize_region(self, region, mask):
        """Normalize a specific region based on its mask"""
        batch, channels, height, width = region.size()

        # Calculate number of pixels in masked region
        num_pixels = mask.sum(dim=(2, 3), keepdim=True)
        num_pixels[num_pixels == 0] = 1  # Avoid division by zero

        # Calculate mean for the region
        region_mean = region.sum(dim=(2, 3), keepdim=True) / num_pixels

        # Normalize with region-specific statistics
        normalized = self.instance_norm(region + (1 - mask) * region_mean)

        # Scale by relative region size
        scale_factor = torch.sqrt(num_pixels / (height * width))
        return normalized * scale_factor

    def forward(self, features, mask):
        mask = mask.detach()

        # Normalize foreground and background separately
        foreground_normalized = self.normalize_region(features * mask, mask)
        background_normalized = self.normalize_region(features * (1 - mask), 1 - mask)

        return foreground_normalized + background_normalized


class SpatiallyAdaptiveNormalization(nn.Module):

    def __init__(self, config, norm_type, num_channels, num_semantic_channels):
        super(SpatiallyAdaptiveNormalization, self).__init__()
        self.config = config
        self.noise_scale = nn.Parameter(torch.zeros(num_channels))

        # Determine base normalization type
        assert norm_type.startswith('alias'), "SPADE norm type must start with 'alias'"
        base_norm_type = norm_type[len('alias'):]

        if base_norm_type == 'batch':
            self.base_norm = nn.BatchNorm2d(num_channels, affine=False)
        elif base_norm_type == 'instance':
            self.base_norm = nn.InstanceNorm2d(num_channels, affine=False)
        elif base_norm_type == 'mask':
            self.base_norm = RegionMaskNormalization(num_channels)
        else:
            raise ValueError(
                f"Unrecognized normalization type in SPADE: {base_norm_type}"
            )

        # Learned modulation parameters
        hidden_channels = 128
        kernel_size = 3
        padding = kernel_size // 2

        self.shared_conv = nn.Sequential(
            nn.Conv2d(num_semantic_channels, hidden_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.gamma_conv = nn.Conv2d(hidden_channels, num_channels,
                                    kernel_size=kernel_size, padding=padding)
        self.beta_conv = nn.Conv2d(hidden_channels, num_channels,
                                   kernel_size=kernel_size, padding=padding)

    def forward(self, features, segmentation, misalignment_mask=None):

        batch, channels, height, width = features.size()

        # Add learned noise for stochasticity
        if self.config.cuda:
            noise = torch.randn(batch, width, height, 1).cuda() * self.noise_scale
        else:
            noise = torch.randn(batch, width, height, 1) * self.noise_scale
        noise = noise.transpose(1, 3)

        # Apply base normalization
        if misalignment_mask is None:
            normalized = self.base_norm(features + noise)
        else:
            normalized = self.base_norm(features + noise, misalignment_mask)

        # Generate affine transformation parameters from segmentation
        activation = self.shared_conv(segmentation)
        gamma = self.gamma_conv(activation)
        beta = self.beta_conv(activation)

        # Apply learned affine transformation
        output = normalized * (1 + gamma) + beta
        return output


class SPADEResidualBlock(nn.Module):
    """Residual block with SPADE normalization"""

    def __init__(self, config, in_channels, out_channels, use_mask_norm=True):
        super(SPADEResidualBlock, self).__init__()
        self.config = config
        self.use_skip_connection = (in_channels != out_channels)
        mid_channels = min(in_channels, out_channels)

        # Main convolution path
        self.conv_main_1 = nn.Conv2d(in_channels, mid_channels,
                                     kernel_size=3, padding=1)
        self.conv_main_2 = nn.Conv2d(mid_channels, out_channels,
                                     kernel_size=3, padding=1)

        # Skip connection
        if self.use_skip_connection:
            self.conv_skip = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1, bias=False)

        # Apply spectral normalization if specified
        norm_type = config.norm_G
        if norm_type.startswith('spectral'):
            norm_type = norm_type[len('spectral'):]
            self.conv_main_1 = spectral_norm(self.conv_main_1)
            self.conv_main_2 = spectral_norm(self.conv_main_2)
            if self.use_skip_connection:
                self.conv_skip = spectral_norm(self.conv_skip)

        # Determine SPADE normalization configuration
        num_semantic_channels = config.gen_semantic_nc
        if use_mask_norm:
            norm_type = 'aliasmask'
            num_semantic_channels += 1

        # SPADE normalization layers
        self.norm_1 = SpatiallyAdaptiveNormalization(
            config, norm_type, in_channels, num_semantic_channels
        )
        self.norm_2 = SpatiallyAdaptiveNormalization(
            config, norm_type, mid_channels, num_semantic_channels
        )
        if self.use_skip_connection:
            self.norm_skip = SpatiallyAdaptiveNormalization(
                config, norm_type, in_channels, num_semantic_channels
            )

        self.activation = nn.LeakyReLU(0.2)

    def apply_skip_connection(self, features, segmentation, misalignment_mask):
        """Apply skip connection with appropriate normalization"""
        if self.use_skip_connection:
            return self.conv_skip(
                self.norm_skip(features, segmentation, misalignment_mask)
            )
        return features

    def forward(self, features, segmentation, misalignment_mask=None):
        """Forward pass through residual block"""
        # Interpolate segmentation to match feature size
        segmentation = F.interpolate(
            segmentation, size=features.size()[2:], mode='nearest'
        )

        if misalignment_mask is not None:
            misalignment_mask = F.interpolate(
                misalignment_mask, size=features.size()[2:], mode='nearest'
            )

        # Skip connection
        skip_features = self.apply_skip_connection(
            features, segmentation, misalignment_mask
        )

        # Main path
        out = self.conv_main_1(
            self.activation(self.norm_1(features, segmentation, misalignment_mask))
        )
        out = self.conv_main_2(
            self.activation(self.norm_2(out, segmentation, misalignment_mask))
        )

        return skip_features + out


class SPADEGenerator(NetworkBase):
    """SPADE-based generator for virtual try-on"""

    def __init__(self, config, input_channels):
        super(SPADEGenerator, self).__init__()
        self.num_upsampling_layers = config.num_upsampling_layers
        self.config = config
        self.spatial_height, self.spatial_width = self._compute_latent_size(config)

        num_filters = config.ngf

        # Initial convolutions at multiple scales
        self.conv_0 = nn.Conv2d(input_channels, num_filters * 16,
                                kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module(
                f'conv_{i}',
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
            )

        # Initial processing block
        self.head_block = SPADEResidualBlock(
            config, num_filters * 16, num_filters * 16, use_mask_norm=False
        )

        # Middle processing blocks
        self.middle_block_1 = SPADEResidualBlock(
            config, num_filters * 16 + 16, num_filters * 16, use_mask_norm=False
        )
        self.middle_block_2 = SPADEResidualBlock(
            config, num_filters * 16 + 16, num_filters * 16, use_mask_norm=False
        )

        # Upsampling blocks
        self.upsample_block_1 = SPADEResidualBlock(
            config, num_filters * 16 + 16, num_filters * 8, use_mask_norm=False
        )
        self.upsample_block_2 = SPADEResidualBlock(
            config, num_filters * 8 + 16, num_filters * 4, use_mask_norm=False
        )
        self.upsample_block_3 = SPADEResidualBlock(
            config, num_filters * 4 + 16, num_filters * 2, use_mask_norm=False
        )
        self.upsample_block_4 = SPADEResidualBlock(
            config, num_filters * 2 + 16, num_filters * 1, use_mask_norm=False
        )

        # Additional upsampling for highest resolution
        if self.num_upsampling_layers == 'most':
            self.upsample_block_5 = SPADEResidualBlock(
                config, num_filters * 1 + 16, num_filters // 2, use_mask_norm=False
            )
            num_filters = num_filters // 2

        # Final output convolution
        self.conv_output = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)

        # Activation functions
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.LeakyReLU(0.2)
        self.output_activation = nn.Tanh()

    def _compute_latent_size(self, config):
        """Calculate spatial dimensions of latent representation"""
        if self.num_upsampling_layers == 'normal':
            num_upsample_ops = 5
        elif self.num_upsampling_layers == 'more':
            num_upsample_ops = 6
        elif self.num_upsampling_layers == 'most':
            num_upsample_ops = 7
        else:
            raise ValueError(
                f"Invalid num_upsampling_layers: {self.num_upsampling_layers}"
            )

        latent_height = config.fine_height // (2 ** num_upsample_ops)
        latent_width = config.fine_width // (2 ** num_upsample_ops)

        return latent_height, latent_width

    def forward(self, input_features, segmentation):
        """Generate output image from input features and segmentation"""
        # Create multi-scale input pyramid
        scale_sizes = [
            (self.spatial_height * (2 ** i), self.spatial_width * (2 ** i))
            for i in range(8)
        ]
        multiscale_inputs = [
            F.interpolate(input_features, size=size, mode='nearest')
            for size in scale_sizes
        ]

        # Extract features at each scale
        multiscale_features = [
            self._modules[f'conv_{i}'](multiscale_inputs[i])
            for i in range(8)
        ]

        # Initial processing
        x = self.head_block(multiscale_features[0], segmentation)
        x = self.upsample(x)

        # Middle blocks
        x = self.middle_block_1(torch.cat([x, multiscale_features[1]], dim=1), segmentation)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.upsample(x)
        x = self.middle_block_2(torch.cat([x, multiscale_features[2]], dim=1), segmentation)

        # Progressive upsampling
        x = self.upsample(x)
        x = self.upsample_block_1(torch.cat([x, multiscale_features[3]], dim=1), segmentation)

        x = self.upsample(x)
        x = self.upsample_block_2(torch.cat([x, multiscale_features[4]], dim=1), segmentation)

        x = self.upsample(x)
        x = self.upsample_block_3(torch.cat([x, multiscale_features[5]], dim=1), segmentation)

        x = self.upsample(x)
        x = self.upsample_block_4(torch.cat([x, multiscale_features[6]], dim=1), segmentation)

        # Highest resolution upsampling if needed
        if self.num_upsampling_layers == 'most':
            x = self.upsample(x)
            x = self.upsample_block_5(torch.cat([x, multiscale_features[7]], dim=1), segmentation)

        # Generate output image
        x = self.conv_output(self.activation(x))
        return self.output_activation(x)


class PatchGANDiscriminator(NetworkBase):
    """
    Multi-layer PatchGAN discriminator
    Classifies overlapping image patches as real or fake
    """

    def __init__(self, config):
        super().__init__()
        self.extract_features = not config.no_ganFeat_loss
        num_filters = config.ndf

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        norm_layer = create_normalization_layer(config.norm_D)

        input_channels = config.gen_semantic_nc + 3

        # Build discriminator layers
        layer_sequence = []

        # First layer (no normalization)
        layer_sequence.append([
            nn.Conv2d(input_channels, num_filters,
                      kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=False)
        ])

        # Intermediate layers
        for layer_idx in range(1, config.n_layers_D):
            prev_filters = num_filters
            num_filters = min(num_filters * 2, 512)
            layer_sequence.append([
                norm_layer(nn.Conv2d(prev_filters, num_filters,
                                     kernel_size=kernel_size, stride=2, padding=padding)),
                nn.LeakyReLU(0.2, inplace=False)
            ])

        # Output layer
        layer_sequence.append([
            nn.Conv2d(num_filters, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ])

        # Create sequential modules for feature extraction
        for idx, layers in enumerate(layer_sequence):
            self.add_module(f'model_{idx}', nn.Sequential(*layers))

    def forward(self, input_tensor):
        """Forward pass through discriminator"""
        intermediate_results = [input_tensor]

        for sub_module in self.children():
            output = sub_module(intermediate_results[-1])
            intermediate_results.append(output)

        # Return intermediate features or just final output
        if self.extract_features:
            return intermediate_results[1:]
        else:
            return intermediate_results[-1]


class MultiscaleDiscriminator(NetworkBase):
    """
    Multi-scale discriminator that operates at different image resolutions
    """

    def __init__(self, config):
        super().__init__()
        self.extract_features = not config.no_ganFeat_loss

        # Create discriminators at different scales
        for scale_idx in range(config.num_D):
            discriminator = PatchGANDiscriminator(config)
            self.add_module(f'discriminator_{scale_idx}', discriminator)

    def downsample_image(self, input_tensor):
        """Downsample input by factor of 2"""
        return F.avg_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=[1, 1],
            count_include_pad=False
        )

    def forward(self, input_tensor):

        all_scale_outputs = []
        current_input = input_tensor

        for name, discriminator in self.named_children():
            outputs = discriminator(current_input)

            if not self.extract_features:
                outputs = [outputs]

            all_scale_outputs.append(outputs)
            current_input = self.downsample_image(current_input)

        return all_scale_outputs


class AdversarialLoss(nn.Module):

    def __init__(self, loss_mode, real_label=1.0, fake_label=0.0,
                 tensor_type=torch.FloatTensor):
        super(AdversarialLoss, self).__init__()
        self.real_label_value = real_label
        self.fake_label_value = fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.tensor_type = tensor_type
        self.loss_mode = loss_mode

        # Validate loss mode
        valid_modes = ['ls', 'original', 'w', 'hinge']
        if loss_mode not in valid_modes:
            raise ValueError(f"Invalid GAN loss mode: {loss_mode}. "
                             f"Choose from {valid_modes}")

    def create_target_tensor(self, input_tensor, is_real_target):
        """Create target label tensor matching input size"""
        if is_real_target:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.tensor_type(1).fill_(
                    self.real_label_value
                )
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input_tensor)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.tensor_type(1).fill_(
                    self.fake_label_value
                )
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input_tensor)

    def create_zero_tensor(self, input_tensor):
        """Create zero tensor matching input size"""
        if self.zero_tensor is None:
            self.zero_tensor = self.tensor_type(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input_tensor)

    def compute_loss(self, prediction, is_real_target, for_discriminator=True):
        """Compute appropriate loss based on loss mode"""

        if self.loss_mode == 'original':
            # Binary cross-entropy loss
            target = self.create_target_tensor(prediction, is_real_target)
            return F.binary_cross_entropy_with_logits(prediction, target)

        elif self.loss_mode == 'ls':
            # Least squares loss
            target = self.create_target_tensor(prediction, is_real_target)
            return F.mse_loss(prediction, target)

        elif self.loss_mode == 'hinge':
            # Hinge loss
            if for_discriminator:
                if is_real_target:
                    # D loss for real: max(0, 1 - D(real))
                    min_val = torch.min(prediction - 1, self.create_zero_tensor(prediction))
                    return -torch.mean(min_val)
                else:
                    # D loss for fake: max(0, 1 + D(fake))
                    min_val = torch.min(-prediction - 1, self.create_zero_tensor(prediction))
                    return -torch.mean(min_val)
            else:
                # Generator always aims for real
                assert is_real_target, "Generator hinge loss must target real"
                return -torch.mean(prediction)

        else:  # Wasserstein loss
            if is_real_target:
                return -prediction.mean()
            else:
                return prediction.mean()

    def __call__(self, prediction, is_real_target, for_discriminator=True):
        # Handle multiscale discriminator outputs
        if isinstance(prediction, list):
            total_loss = 0

            for pred_scale in prediction:
                # Extract final prediction if nested list
                if isinstance(pred_scale, list):
                    pred_scale = pred_scale[-1]

                # Compute loss for this scale
                loss_tensor = self.compute_loss(
                    pred_scale, is_real_target, for_discriminator
                )

                # Average across batch
                batch_size = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                scale_loss = torch.mean(loss_tensor.view(batch_size, -1), dim=1)
                total_loss += scale_loss

            return total_loss / len(prediction)
        else:
            return self.compute_loss(prediction, is_real_target, for_discriminator)


def create_normalization_layer(norm_type='instance'):

    def get_output_channels(layer):
        """Extract output channels from layer"""
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_normalization(layer):
        """Add normalization to layer"""
        nonlocal norm_type

        # Apply spectral normalization if specified
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            norm_type = norm_type[len('spectral'):]

        # No additional normalization needed
        if norm_type == 'none' or len(norm_type) == 0:
            return layer

        # Remove bias since it's redundant after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        # Create appropriate normalization layer
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_output_channels(layer), affine=True)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_output_channels(layer), affine=False)
        else:
            raise ValueError(f"Unrecognized normalization type: {norm_type}")

        return nn.Sequential(layer, norm_layer)

    return add_normalization