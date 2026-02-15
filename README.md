## Accelerated Sequential Flow Matching: A Bayesian Filtering Perspective

This repository implements Sequential Flow Matching for the paper **Accelerated Sequential Flow Matching: A Bayesian Filtering Perspective** (https://arxiv.org/abs/2602.05319).

The code builds upon the codebase of Diffusion Forcing (https://github.com/buoyancy99/diffusion-forcing). We thank the author for making the code publicly available.


### Overview
We introduce Sequential Flow Matching, a principled framework grounded in Bayesian filtering for streaming inference problems. It formalizes streaming inference as learning a probability flow that transports the predictive distribution from one time step to the next, rather than sampling from non-informative Gaussian initial distribution per time step. 
This offers a principled warm start that can accelerate sampling compared to the naïve re-sampling.

**Pipeline:** Overall, we obtain a Sequential Flow Matching model via three steps: (1) *model pretraining*: we first use an offline dataset to pretrain a regular diffusion-/flow-based model;
(2) *finetuning dataset generation*: the pretrained model will sample and store a few generated trajectories as the finetuning dataset (whose size is much smaller than pretraining dataset);
(3) *model finetuning*: the pretrained model will be finetuned on the finetuning dataset. The final finetuned model is a flow matching model that explicitly learns how to transport the predictive distribution from current time step to next time step.



### Installation
```
pip install -r requirements.txt
```
For D4RL benchmark, we additionally need:
```
pip install -r extra_requirements.txt
```

Alternatively, please check ``df_v2.yaml`` for the full conda environment file.


### Pretraining Dataset Download 

#### 1. 1D Burgers' Equation Online Forecasting

We adopt the dataset provided by https://github.com/AI4Science-WestlakeU/CL_DiffPhyCon. 
Please visit https://drive.google.com/drive/folders/1moLdtqmvmAU8FoWt6ELWOTXT0tPuY-qJ, and download ``CL-DiffPhyCon/Train_dataset/1D/train_data`` and ``CL-DiffPhyCon/Test_dataset/1D/test_data`` from there, 
and put them under ``./data/pde/`` in this repository.

#### 2. Weather Forecasting

We adopt WeatherBench2 (https://github.com/google-research/weatherbench2). The dataset can be automatically accessed via their API by installing weatherbench2 python package.

#### 3. Maze Planning

We adopt maze planning tasks from D4RL (https://github.com/Farama-Foundation/D4RL). The dataset will be automatically downloaded via the code.

#### 4. Smoke Control

We adopt the dataset again from https://github.com/AI4Science-WestlakeU/CL_DiffPhyCon. 
Please visit https://drive.google.com/drive/folders/1moLdtqmvmAU8FoWt6ELWOTXT0tPuY-qJ, and download ``CL-DiffPhyCon/Train_dataset/2D/train_x0000-y0000.zip`` and ``CL-DiffPhyCon/Test_dataset/1D/test.zip`` from there, 
and unzip them under ``./data/smoke/`` in this repository. 





### Step1: Model Pretraining

This stage train a diffusion-/flow-based model for sequence generation, with standard Gaussian as the initial distribution. We adopt DiffusionForcing (https://github.com/buoyancy99/diffusion-forcing) types of training which applies random noise levels to different tokens during training. 

For Burgers' equation forecasting, run
```
python main.py +name=burger_flow_pretrain experiment=exp_simulation experiment.tasks=[training] algorithm=flow_simulation dataset=simulation_burgers dataset.control_dim=0
```
For weather forecasting, run 
```
python main +name=weather_flow_pretrain experiment.tasks=[training] experiment=exp_weather algorithm=flow_weather dataset=weather 
```
For maze planning, we directly adopt an existing checkpoint of DiffusionForcing model for maze planning from https://github.com/buoyancy99/diffusion-forcing,
and the checkpoint is already saved at `./outputs/maze2d_medium_x_new.ckpt`.

For smoke control, run
```
python main.py +name=pde2d_df_pretrain experiment=exp_pde algorithm=df_pde_2D dataset=pde_smoke dataset.save_dir=data/smoke 
```

After pretraining, you will have a checkpoint `CKPT` (which can be a wandb run id). We use ``CKPT`` to represent the checkpoint (it is different for different tasks) and it will then be used in the rest procedure. 


### Step 2: Finetuning Dataset Generation

For Burgers' equation forecasting, run
```
python main.py +name=burger_save experiment=exp_simulation experiment.tasks=[save_inference] \
algorithm=flow_simulation_finetuner dataset=simulation_burgers load=CKPT \
algorithm.num_gen_trials=3 dataset.control_dim=0
```

For Weather forecasting, run
```
python main +name=weather_save experiment.tasks=[save_inference] experiment=exp_weather algorithm=flow_weather_finetuner \
load=CKPT dataset=weather 
```

For maze planning, run
```
python main.py +name=maze_save experiment=exp_planning \
experiment.tasks=[save_inference] algorithm=df_planning_finetuner dataset=maze2d_medium \
dataset.action_mean=[] dataset.action_std=[] \
dataset.observation_mean=[3.5092521,3.4765592] dataset.observation_std=[1.3371079,1.52102] \
algorithm.guidance_scale=3 \
load=outputs/maze2d_medium_x_new.ckpt \
experiment.training.batch_size=4 experiment.validation.batch_size=4 experiment.validation.limit_batch=1000 \
algorithm.finetune_external_cond_dim=0 \
algorithm.num_gen_trials=5
```

For smoke control, run
```
python main.py +name=smoke_save dataset.finetune_sim_range=[36000,38000] experiment=exp_pde \
algorithm=df_pde_2D_finetuner dataset=pde_smoke experiment.tasks=[save_inference] \
load=CKPT dataset.save_dir=data/smoke \
experiment.training.batch_size=16 experiment.validation.limit_batch=999 algorithm.diffusion.sampling_timesteps=10 \
experiment.validation.batch_size=16
```


### Step 3: Finetuning

After saving the finetuning dataset, we can finetune the pretrained model to directly learn the Bayesian filtering update, i.e., sequential flow model.

For Burger's equation forecasting, run
```
python main.py +name=burger_finetune experiment=exp_simulation dataset=simulation_burgers \
algorithm=flow_simulation_finetuner experiment.tasks=[fine_tune] load=CKPT experiment.training.max_steps=4005 \
dataset.control_dim=0 algorithm.num_gen_trials=10 algorithm.update_sampling_timesteps=1 algorithm.renoise_level=400 \
algorithm.flow_time=400 algorithm.lr=5e-5
```

For weather forecasting, run
```
python main.py +name=weather_finetune experiment.tasks=[fine_tune] experiment=exp_weather \
algorithm=flow_weather_finetuner dataset=weather algorithm.n_fine_tuning_data=5000 \ 
experiment.training.max_steps=400000 algorithm.lr=8e-4 algorithm.num_gen_trials=5 \
experiment.validation.limit_batch=5 experiment.validation.batch_size=6 \
algorithm.warmup_steps=1000 algorithm.renoise_level=200 algorithm.flow_time=200 \
load=CKPT experiment.validation.val_every_n_step=2000
```

For maze planning, run
```
python main.py +name=maze_finetune experiment=exp_planning experiment.tasks=[fine_tune] \
algorithm=df_planning_finetuner dataset=maze2d_medium dataset.action_mean=[] dataset.action_std=[] \
dataset.observation_mean=[3.5092521,3.4765592] dataset.observation_std=[1.3371079,1.52102] \
load=outputs/maze2d_medium_x_new.ckpt algorithm.guidance_scale=3 \
experiment.training.batch_size=1024 experiment.training.max_steps=100000
```


For smoke control, run
```
python main.py +name=pde2d_finetune dataset.finetune_sim_range=[36000,38000] \
experiment=exp_pde algorithm=df_pde_2D_finetuner dataset=pde_smoke experiment.tasks=[fine_tune] \
load=CKPT dataset.save_dir=data/smoke \
experiment.training.max_steps=40005 experiment.training.checkpointing.every_n_train_steps=10000 \
experiment.training.batch_size=16 experiment.validation.batch_size=8 experiment.validation.val_every_n_step=500 \
experiment.validation.limit_batch=1 algorithm.lr=4e-5 \
```

After finetuning, we can evaluate it using the same finetuning command but with ``experiment.tasks=[validation]``.