## DMotion
### Paper

Haoqi Yuan, Ruihai Wu, Andrew Zhao, Haipeng Zhang, Zihan Ding, Hao Dong, ["DMotion: Robotic Visuomotor Control with Unsupervised Forward Model Learned from Videos"](https://arxiv.org/abs/2103.04301), IROS 2021

[arXiv](https://arxiv.org/abs/2103.04301) 

[project page](https://hyperplane-lab.github.io/dmotion/)

## Experiments in Grid World, Atari Pong

### Dependencies:
- Ubuntu 
- python 3.7
- dependencies: PIL, numpy, tqdm, gym, pytorch (1.2.0), matplotlib


### (a). Generate datasets:
Grid World: 

`python datagen/gridworld.py`

Atari Pong:  require gym[atari] to be installed. 

`python datagen/ataripong.py `


### (b). Train the model from unlabelled videos:

Grid World: 

`python train.py --gpu 0 --dataset_name shape --data_path datagen/gridworld`

Atari Pong: 

`python train.py --gpu 0 --dataset_name pong --data_path datagen/atari-pong`

Set `--gpu -1` if use cpu. 

### (c). Find action-transformation mapping and test:
One transition sample for each action can be manually selected. In folder `datagen/demonstration`, 
we have one directory for each dataset to contain the demonstrations. Directory names are `shape`, `pong`, with respect to two datasets.

We provide you the demonstrations for all datasets. For example, for Grid World dataset, directory `datagen/demonstration/shape` contains a file `demo.txt`, in which each line consists of {img1, img2, action}, showing a transition of this action. Image files img1, img2 are placed in directory `datagen/demonstration/shape`. 

 You can either use the demonstrations provided by us, or arbitrarily replace them with samples in the generated dataset manually. 


Grid World: 

`python test.py --gpu 0 --dataset_name shape --data_path datagen/gridworld`

Atari Pong: 

`python test.py --gpu 0 --dataset_name pong --data_path datagen/atari-pong`

Set `--gpu -1` if use cpu. You can select the model for test using e.g., `--test_model_path checkpoint/model_49.pth`. 
The test program will sequentially run the test of feature map visualisation, visual forecasting conditioned on the agent's motion and quantitative test. If you want to disable them, add `--visualize_feature 0`, `--visual_forecasting 0` or `--quantitative 0`, respectively.


## Experiments in Robot Pushing and MPC

### Dependencies:

Please visit https://github.com/stepjam/PyRep, and follow the installation on their page to install Pyrep and CoppeliaSim.

### Generate Dataset and Train:

1. Use `python -m datagen.roboarm --increment 20` to generate trajectories, as the simulation gets slower, please only generate around 20-100 (--increment) trajectories at a time. When generating a second batch, please use `python -m datagen.roboarm --increment 20 --eps [number of eps generated before + increment]` for the sake of naming convention.
2. [Optional] If generated multiple batches of trajectories, use `python -m rename --folders [number of folders you generated in step 1]` to move all files into a single folder.
3. Train: `python -m train --dataset_name roboarm --data_path datagen/roboarm --size 64 --no_graph --contrastive_coeff 0 --save_path checkpoint`
4. Test: `python -m test --dataset_name roboarm --data_path datagen/roboarm --size 64 --no_graph --contrastive_coeff 0 --test_model_path checkpoint/model_49.pth --test_save_path test`

### Model Predictive Control

** Generate data for MPC**

Create csv file logging the trajectory IDs and time step you wish to perform MPC on  


   | episode | start_idx | end_idx |
   | ------- | --------- | ------- |
   | 200     | 0         | 29      |
   | 201     | 8         | 37      |

Run `python -m rename_mpc --folders 1` to combine all json state files into one for MPC dataset. Change the `--folders` argument according to the maximum ID of the trajectories manually selected. Run `python -m move_mpc` to get all trajectories to the right place.

** Test for MPC **

 `python -m mpc_multi`
 
 
## Citation

If you find this code useful for your research, please cite our paper:

```
@article{yuan2021robotic,
  title={Robotic Visuomotor Control with Unsupervised Forward Model Learned from Videos},
  author={Yuan, Haoqi and Wu, Ruihai and Zhao, Andrew and Zhang, Haipeng and Ding, Zihan and Dong, Hao},
  journal={arXiv preprint arXiv:2103.04301},
  year={2021}
}
```