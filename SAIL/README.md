### Train VAE
```bash
python train_VAE.py --expert-traj-path /home/$USER/data/demonstrations.pkl --state-dim 3 --output-path /home/$USER/data --size-per-traj 200
```
`demonstrations.pkl` should be a pickle containing a list of numpy arrays. Each numpy array is an expert trajectory (s_1, s_2, ..., s_t, s_t+1, .., s_N) with shape of (N, state_dim). N is the total number of steps. which is a fixed number specified in the argument `--size-per-traj`. state_dime is the dimension of state space.

### Test VAE
```python
    model = VAE(state_dim)
    model_dict = torch.load('vae.pt', map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval

```

# State Alignment-based Imitation Learning
We propose a state-based imitation learning method for cross-morphology imitation learning, by considering both the state visitation distribution and local transition alignment.

### ATTENTION: The code is in a pre-release stage and under maintenance. Some details may be different from the paper or not included, due to internal integration for other projects. The author will reorganize some parts to match the original paper asap.

## Train
The demonstration is provided in ``./expert/assets/expert_traj``. They are obtained from well-trained expert algorithms (SAC or TRPO).

The cross-morphology imitator can be founded in ``./envs/mujoco/assets``. You can also customize your own environment for training.

The main algorithm contains two settings: original imitation learning and cross-morphology imitation learning. 
The only difference is the pretraining stage for the inverse dynamics model.

The format should be
```bash
python sail.py --env-name [YOUR-ENV-NAME] --expert-traj-path [PATH-TO-DEMO] --beta 0.01 --resume [if want resume] --transfer [if cross morphology]
```
For example, for original hopper imitation:
```bash
python sail.py --env-name Hopper-v2 --expert-traj-path ./expert/assets/expert_traj/Hopper-v2_expert_traj.p --beta 0.005
```
for disabled swimmer imitation
```bash
python sail.py --env-name DisableSwimmer-v0 --expert-traj-path ./expert/assets/expert_traj/Swimmer-v2_expert_traj.p --beta 0.005 --transfer
```

## Cite Our Paper
If you find it useful, please consider to cite our paper.
```
@article{liu2019state,
  title={State Alignment-based Imitation Learning},
  author={Liu, Fangchen and Ling, Zhan and Mu, Tongzhou and Su, Hao},
  journal={arXiv preprint arXiv:1911.10947},
  year={2019}
}
```

## Demonstrations
Please download [here](https://drive.google.com/open?id=1cIqYevPDE2_06Elyo_UqEfHiQvOKP6in) and put it to ```./expert/assets/expert_traj```
