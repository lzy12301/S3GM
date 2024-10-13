import argparse


# Parameters
parser = argparse.ArgumentParser(description='S3GM demo')
parser.add_argument('--data', type=str, default='kolmogorov', help='kse, kolmogorov, cylinder, era5')
parser.add_argument('--dataset', type=str, default='DatasetNS4KVideo', help='DatasetNS4RDVideo, DatasetNS4KVideo')
parser.add_argument('--data_location', type=str, default="../data/kolmogorov_vel_train.npy", help='path for training data')
parser.add_argument('--version', type=str, default='v0', help='model version')
parser.add_argument('--continue_training', type=int, default=0, help='continue training or not')
parser.add_argument('--num_samples_train', type=int, default=100000, help='number of training samples')
parser.add_argument('--num_samples_val', type=int, default=1000, help='number of evaluating samples')
parser.add_argument('--train_split', type=float, default=0.9, help='split the training and evaluating data')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epochs', type=int, default=700)
parser.add_argument('--lr', type=float, default=2e-4, help='ns1: 2e-4, ns2: 2e-5, fno: 5e-5')
parser.add_argument('--batch_size', type=int, default=32, help='2, 4096(cifar10), 128(ns1), 4(ns2)')
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--checkpoints_path', type=str, default='checkpoints/')

# data
parser.add_argument('--image_size', type=int, default=64, help='1024 for kse, 64 for others')
parser.add_argument('--dims', type=int, default=2, help='Spatial dimension of the system')
parser.add_argument('--num_components', type=int, default=2, help='number of components for state variable')
parser.add_argument('--num_conditions', type=int, default=2, help='number of parameters')
parser.add_argument('--num_frames', type=int, default=10, help='number of frames for dynamical systems')
parser.add_argument('--num_interval', type=int, default=1, help='temporal down-sample scale')
parser.add_argument('--is_scalar', type=int, default=1, help='continue training or not')

# model
parser.add_argument('--num_scales', type=int, default=1000)
parser.add_argument('--beta_min', type=float, default=0.1)
parser.add_argument('--beta_max', type=float, default=20.)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--ema_rate', type=float, default=0.999, help='ns1: 0.999, ns2: 0.9999')
parser.add_argument('--nf', type=int, default=32, help='128 for all, 16 for autoregressive')
parser.add_argument('--ch_mult', type=int, nargs='+', default=(1, 2, 4, 8),
                    help='ns1: (1, 2, 2, 2), '
                         'ns2: (1, 1, 2, 2, 4, 4)')
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--attn_resolutions', type=int, nargs='+', default=(8, 16,))
parser.add_argument('--num_heads', type=int, default=1)

# training
parser.add_argument('--continuous', type=int, default=1)

# sampling
parser.add_argument('--predictor', type=str, default='reverse_diffusion')
parser.add_argument('--corrector', type=str, default='langevin')
parser.add_argument('--mm_mode', type=list, default=[0, 0, 0], help='[0, 1] for ns3_mm, [0, 0, 0] for ns4')
parser.add_argument('--N', type=int, default=200)
parser.add_argument('--inner_loop', type=int, default=5)
parser.add_argument('--num_samples', type=int, default=1, help='number of samples to be generated using PFGM')
parser.add_argument('--is_forward', type=bool, default=True,
                    help='whether to do forward prediction or inverse prediction')
parser.add_argument('--physics_guide', type=bool, default=False,
                    help='whether to use physics-guided generation during sampling')
parser.add_argument('--lamb', type=float, default=1e0, help='step size for physics-guided gradient descent')
parser.add_argument('--num_steps', type=int, default=1, help='number of steps for forward/inverse prediction')
parser.add_argument('--overlap', type=int, default=1, help='overlaps for forward/inverse prediction')
parser.add_argument('--obs_type', type=str, default='pred', help='pred, phys, spec, para')

config = parser.parse_args()  # args=[]
