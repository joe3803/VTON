import argparse
import os
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.metrics import structural_similarity as calculate_ssim
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.models.inception import inception_v3
import eval_models as models


class EvaluationConfig:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Virtual Try-On Model Evaluation')
        parser.add_argument('--evaluation', default='LPIPS',
                            help='Evaluation metric type')
        parser.add_argument('--predict_dir', default='./result/bg_ver1/output/',
                            help='Directory containing predicted images')
        parser.add_argument('--ground_truth_dir', default='./data/zalando-hd-resize/test/image',
                            help='Directory containing ground truth images')
        parser.add_argument('--resolution', type=int, default=1024,
                            help='Image resolution for evaluation')

        self.args = parser.parse_args()

    def __getattr__(self, name):
        return getattr(self.args, name)


class ImageTransforms:
    """Pre-defined image transformations for different metrics"""

    @staticmethod
    def get_standard_transform():
        """Standard transform: convert to tensor only"""
        return transforms.ToTensor()

    @staticmethod
    def get_lpips_transform():
        """Transform for LPIPS metric: 128x128 with normalization"""
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    @staticmethod
    def get_inception_transform():
        """Transform for Inception Score: 299x299 with normalization"""
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


class MetricCalculator:
    """Handles computation of various image quality metrics"""

    def __init__(self, config):
        self.config = config
        self.transforms = ImageTransforms()

        # Initialize models
        self.lpips_model = models.PerceptualLoss(
            model='net-lin',
            net='alex',
            use_gpu=True
        )
        self.lpips_model.eval()

        self.inception_model = inception_v3(
            pretrained=True,
            transform_input=False
        ).type(torch.cuda.FloatTensor)
        self.inception_model.eval()

        # Metric accumulators
        self.ssim_sum = 0.0
        self.mse_sum = 0.0
        self.lpips_sum = 0.0
        self.lpips_scores = []
        self.inception_predictions = None

    def resize_ground_truth(self, gt_image):
        """Resize ground truth image based on resolution setting"""
        if self.config.resolution == 1024:
            return gt_image
        elif self.config.resolution == 512:
            return gt_image.resize((384, 512), Image.BILINEAR)
        elif self.config.resolution == 256:
            return gt_image.resize((192, 256), Image.BILINEAR)
        else:
            raise NotImplementedError(f"Resolution {self.config.resolution} not supported")

    def compute_ssim(self, gt_image, pred_image):
        """Compute Structural Similarity Index"""
        gt_gray = np.asarray(gt_image.convert('L'))
        pred_gray = np.asarray(pred_image.convert('L'))

        return calculate_ssim(
            gt_gray,
            pred_gray,
            data_range=255,
            gaussian_weights=True,
            use_sample_covariance=False
        )

    def compute_lpips(self, gt_image, pred_image):
        transform = self.transforms.get_lpips_transform()

        gt_tensor = transform(gt_image).unsqueeze(0).cuda()
        pred_tensor = transform(pred_image).unsqueeze(0).cuda()

        return self.lpips_model.forward(gt_tensor, pred_tensor).item()

    def compute_mse(self, gt_image, pred_image):
        transform = self.transforms.get_standard_transform()

        gt_tensor = transform(gt_image).unsqueeze(0).cuda()
        pred_tensor = transform(pred_image).unsqueeze(0).cuda()

        return F.mse_loss(gt_tensor, pred_tensor).item()

    def get_inception_prediction(self, pred_image):
        transform = self.transforms.get_inception_transform()

        pred_tensor = transform(pred_image).unsqueeze(0).cuda()
        prediction = F.softmax(self.inception_model(pred_tensor), dim=1)

        return prediction.data.cpu().numpy()

    def compute_inception_score(self, num_images, num_splits=1):
        split_scores = []

        for split_idx in range(num_splits):
            start_idx = split_idx * (num_images // num_splits)
            end_idx = (split_idx + 1) * (num_images // num_splits)
            split_predictions = self.inception_predictions[start_idx:end_idx, :]

            # Marginal distribution
            marginal_dist = np.mean(split_predictions, axis=0)

            # KL divergence for each prediction
            kl_divergences = []
            for i in range(split_predictions.shape[0]):
                conditional_dist = split_predictions[i, :]
                kl_divergences.append(entropy(conditional_dist, marginal_dist))

            split_scores.append(np.exp(np.mean(kl_divergences)))

        return np.mean(split_scores), np.std(split_scores)

    def save_lpips_rankings(self, output_dir):
        """Save LPIPS scores sorted from worst to best"""
        self.lpips_scores.sort(key=lambda x: x[1], reverse=True)

        lpips_file = os.path.join(output_dir, 'lpips_rankings.txt')
        with open(lpips_file, 'w') as f:
            for img_name, score in self.lpips_scores:
                f.write(f"{img_name} {score}\n")


class VirtualTryOnEvaluator:
    def __init__(self, config):
        self.config = config
        self.calculator = MetricCalculator(config)

    def load_image_lists(self):
        pred_images = os.listdir(self.config.predict_dir)
        gt_images = os.listdir(self.config.ground_truth_dir)

        pred_images.sort()
        gt_images.sort()

        return pred_images, gt_images

    def get_ground_truth_name(self, pred_name):
        """Extract ground truth filename from prediction filename"""
        return pred_name.split('_')[0] + '_00.jpg'

    def evaluate(self, pred_list, gt_list):

        num_images = len(gt_list)
        self.calculator.inception_predictions = np.zeros((num_images, 1000))

        print("=" * 60)
        print("Starting Evaluation: SSIM, MSE, LPIPS, Inception Score")
        print("=" * 60)

        with torch.no_grad():
            for idx, pred_filename in enumerate(pred_list):
                gt_filename = self.get_ground_truth_name(pred_filename)

                # Load images
                gt_path = os.path.join(self.config.ground_truth_dir, gt_filename)
                pred_path = os.path.join(self.config.predict_dir, pred_filename)

                gt_image = Image.open(gt_path)
                gt_image = self.calculator.resize_ground_truth(gt_image)
                pred_image = Image.open(pred_path)

                # Verify image sizes match
                assert gt_image.size == pred_image.size, \
                    f"Size mismatch: GT={gt_image.size} vs Pred={pred_image.size}"

                # Compute all metrics
                ssim_score = self.calculator.compute_ssim(gt_image, pred_image)
                self.calculator.ssim_sum += ssim_score

                lpips_score = self.calculator.compute_lpips(gt_image, pred_image)
                self.calculator.lpips_sum += lpips_score
                self.calculator.lpips_scores.append((pred_filename, lpips_score))

                mse_score = self.calculator.compute_mse(gt_image, pred_image)
                self.calculator.mse_sum += mse_score

                inception_pred = self.calculator.get_inception_prediction(pred_image)
                self.calculator.inception_predictions[idx] = inception_pred

                # Progress update
                if (idx + 1) % 10 == 0 or idx == 0:
                    print(f"Progress: {idx + 1}/{num_images} | "
                          f"LPIPS: {lpips_score:.4f} | "
                          f"SSIM: {ssim_score:.4f}")

        # Compute averages
        avg_ssim = self.calculator.ssim_sum / num_images
        avg_mse = self.calculator.mse_sum / num_images
        avg_lpips = self.calculator.lpips_sum / num_images

        # Compute Inception Score
        print("\n" + "=" * 60)
        print("Computing Inception Score...")
        print("=" * 60)
        is_mean, is_std = self.calculator.compute_inception_score(num_images)

        # Save detailed results
        self.calculator.save_lpips_rankings(self.config.predict_dir)
        self.save_summary_results(avg_ssim, avg_mse, avg_lpips, is_mean, is_std)

        return {
            'ssim': avg_ssim,
            'mse': avg_mse,
            'lpips': avg_lpips,
            'inception_mean': is_mean,
            'inception_std': is_std
        }

    def save_summary_results(self, ssim, mse, lpips, is_mean, is_std):
        """Save evaluation summary to file"""
        summary_file = os.path.join(self.config.predict_dir, 'evaluation_summary.txt')

        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Virtual Try-On Model Evaluation Results\n")
            f.write("=" * 60 + "\n\n")

            f.write("Image Quality Metrics:\n")
            f.write(f"  SSIM (Structural Similarity):  {ssim:.6f}\n")
            f.write(f"  MSE (Mean Squared Error):      {mse:.6f}\n")
            f.write(f"  LPIPS (Perceptual Distance):   {lpips:.6f}\n\n")

            f.write("Generative Quality Metrics:\n")
            f.write(f"  Inception Score (Mean):        {is_mean:.6f}\n")
            f.write(f"  Inception Score (Std):         {is_std:.6f}\n")
            f.write("=" * 60 + "\n")

    def print_results(self, results):
        """Print formatted results to console"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"SSIM:            {results['ssim']:.6f}")
        print(f"MSE:             {results['mse']:.6f}")
        print(f"LPIPS:           {results['lpips']:.6f}")
        print(f"Inception Score: {results['inception_mean']:.6f} ± {results['inception_std']:.6f}")
        print("=" * 60 + "\n")


def main():
    """Main evaluation pipeline"""
    # Initialize configuration
    config = EvaluationConfig()

    # Create evaluator
    evaluator = VirtualTryOnEvaluator(config)

    # Load image lists
    pred_list, gt_list = evaluator.load_image_lists()

    print(f"\nFound {len(pred_list)} prediction images")
    print(f"Found {len(gt_list)} ground truth images\n")

    # Run evaluation
    results = evaluator.evaluate(pred_list, gt_list)

    # Display results
    evaluator.print_results(results)

    print(f"Detailed results saved to: {config.predict_dir}")


if __name__ == '__main__':
    main()