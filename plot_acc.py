import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from config import parser

args = parser.parse_args()


def make_image(file_path, label, color, linestyle, data_type):
	steps = np.arange(0, args.total_steps + 1)
	if data_type == 'state':
    		file_name = 'acc_gt_pos_err'
	else:
    		file_name = 'acc_pos_err'
	files = glob(f'{file_path}/{file_name}*.npy')
	result = np.stack([np.load(f).squeeze() for f in files], axis=0)
	normalizer = np.expand_dims(result[:, 0], 1)
	result = result / normalizer
	aver_result = np.mean(result, axis=0)
	std_result = np.std(result, axis=0)
	print(file_path.split('/')[0], aver_result[-1], 0.5*std_result[-1])
	ax.plot(steps, aver_result, label=label, color=color, linestyle=linestyle)
	ax.fill_between(steps, aver_result-0.5 * std_result, aver_result+0.5 * std_result,
					alpha=0.1, color=color)

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
data_type = 'state'

# EDCNN
file_path = f'test_mpc_1_10_edcnn/4_5_50_{args.total_steps}_euc/acc'
make_image(file_path, label='E-D CNN  (10%)', color='blue', linestyle='--', data_type=data_type)
file_path = f'test_mpc_edcnn/4_5_50_{args.total_steps}_euc/acc'
make_image(file_path, label='E-D CNN', color='blue', linestyle='-', data_type=data_type)

# MOOD
file_path = f'test_mpc_mood/4_5_50_{args.total_steps}_euc/acc'
make_image(file_path, label='Ours', color='red', linestyle='-', data_type=data_type)
# file_path = f'test_mpc_mood/4_5_50_{args.total_steps}_manhattan/acc'
# make_image(file_path, label='Ours (Manhattan)', color='red', linestyle='-', data_type=data_type)

# WMAE
file_path = f'test_mpc_wmaefcn/4_5_50_{args.total_steps}_euc/acc'
make_image(file_path, label='WM AE', color='green', linestyle='-', data_type=data_type)
file_path = f'test_mpc_1_10_wmaefcn/4_5_50_{args.total_steps}_euc/acc'
make_image(file_path, label='WM AE  (10%)', color='green', linestyle='--', data_type=data_type)

# CLASP
file_path = f'test_mpc_clasp/4_5_50_{args.total_steps}_euc/acc'
make_image(file_path, label='CLASP', color='black', linestyle='-', data_type=data_type)


ax.legend(loc=3)
plt.xlabel('Steps')
plt.ylabel('Normalized Distance')
plt.grid(linestyle='--')
# plt.title('Model predictive control results')
plt.savefig(f'result_{data_type}_euc.png', dpi=400, bbox_inches='tight')
plt.show()
