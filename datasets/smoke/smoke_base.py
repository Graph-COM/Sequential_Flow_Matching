import numpy as np
import torch
from torch.utils.data import Dataset
import pdb
import sys, os
from pathlib import Path
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


class SmokeDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig, 
        split: str = "training",
    ):
        super().__init__()
        self.cfg = cfg
        if split == "training":
            self.save_dir = Path(cfg.save_dir) / "train"
            self.is_train = True
        elif split == 'finetune':
            self.save_dir = Path(cfg.save_dir) / "train"
            self.is_train = False
        elif split in ["validation", "test"]:
            self.save_dir = Path(cfg.save_dir) / "test"
            self.is_train = False
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split

        self.episode_len = cfg.episode_len
        # self.time_steps = time_steps # total time steps of each trajectory after down sampling
        self.horizon = cfg.horizon # horizon of diffusion model
        self.frame_skip = cfg.frame_skip
        self.time_steps_effective = (self.episode_len - self.horizon + 1) // self.frame_skip
        self.true_size = cfg.true_size
        self.resolution = cfg.resolution
        self.space_interval = int(cfg.true_size/cfg.resolution)

        if split == "training":
            self.n_simu = 36000  # sim_ids 0-35999 (inclusive)
        elif split == "finetune":
            # Check if finetune_sim_range is specified in config
            finetune_range = cfg.get('finetune_sim_range', None)
            if finetune_range is not None:
                # finetune_range should be [start, end] where both are inclusive
                self.finetune_sim_start = finetune_range[0]
                self.finetune_sim_end = finetune_range[1]
                # Validate range
                if self.finetune_sim_start < 0:
                    raise ValueError(f"finetune_sim_range start must be >= 0, got {self.finetune_sim_start}")
                if self.finetune_sim_end < self.finetune_sim_start:
                    raise ValueError(f"finetune_sim_range end ({self.finetune_sim_end}) must be >= start ({self.finetune_sim_start})")
                self.n_simu = self.finetune_sim_end - self.finetune_sim_start + 1
                print(f"Finetune mode: using sim_ids from {self.finetune_sim_start} to {self.finetune_sim_end} (inclusive), total: {self.n_simu} simulations")
            else:
                raise ValueError(f"finetune_sim_range not specified in config for finetune split")
        else:
            self.n_simu = 50
        # self.RESCALER = torch.tensor([3, 20, 20, 17, 19, 1]).reshape(1, 6, 1, 1) 
        self.RESCALER = torch.tensor(cfg.rescaler).reshape(1, 6, 1, 1)  # rescale the data to [-1, 1] with relaxation, on 64 time steps dataset

    def __len__(self):
        # return self.n_simu
        if self.is_train:
            return self.n_simu * self.time_steps_effective
        else:
            return self.n_simu

    def __getitem__(self, idx):
        if self.is_train:
            sim_id, time_id = divmod(idx, self.time_steps_effective)
        else:
            # For finetune split, map idx to actual sim_id using the specified range
            if self.split == 'finetune':
                sim_id = self.finetune_sim_start + idx
                time_id = 0
            else:
                sim_id, time_id = idx, 0 # for test, pass each trajectory as a whole and only once

        if self.is_train:
            # [1, 65, 64, 64]
            density = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            # [2, 65, 64, 64]
            velocity = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1) # 2, 65, 64, 64
            # [2, 65, 64, 64]
            control = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            
            # Shift control one step behind in the time dimension (dim=1)
            # Pad a zero control at the beginning and remove the last frame (so control shifts one step back)
            control_shape = control.shape  # (2, 65, 64, 64)
            zero_control = torch.zeros(control_shape[0], 1, control_shape[2], control_shape[3], dtype=control.dtype, device=control.device)
            control = torch.cat([zero_control, control[:, :-1]], dim=1)  # (2, 65, 64, 64)

            # [65, 8]
            smoke = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float) # 65, 8
            smoke = smoke[:, 1]/smoke.sum(-1) # shape: [65]; 1 is index of of the target bucket
             # [1, 65, 64, 64]
            smoke = smoke.reshape(1, smoke.shape[0], 1, 1).expand(1, smoke.shape[0], self.resolution, self.resolution) # 1, 65, 64, 64


            # [6, 10, 64, 64]
            state = torch.cat((density, velocity, smoke, control), dim=0)[:, time_id: time_id + self.horizon] # 6, horizon, 64, 64
            # rescale: density/1 velocity_x/45 velocity_y/50 smoke/1 control_x/45 control_y/50 
            data = (
                state.permute(1, 0, 2, 3) / self.RESCALER, # horizon, 6, 64, 64
                sim_id,
            )
        else:
            density = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            velocity = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            control = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            
            # Shift control one step behind in the time dimension (dim=1)
            # Pad a zero control at the beginning and remove the last frame (so control shifts one step back)
            control_shape = control.shape  # (2, 65, 64, 64)
            zero_control = torch.zeros(control_shape[0], 1, control_shape[2], control_shape[3], dtype=control.dtype, device=control.device)
            control = torch.cat([zero_control, control[:, :-1]], dim=1)  # (2, 65, 64, 64)

            smoke = torch.tensor(np.load(os.path.join(self.save_dir, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float)
            
            smoke = smoke[:, 1]/smoke.sum(-1)
            smoke = smoke.reshape(1, smoke.shape[0], 1, 1).expand(1, smoke.shape[0], self.resolution, self.resolution) 

            state = torch.cat((density, velocity, smoke, control), dim=0) # 6, 65, 64, 64
            data = (
                # state.permute(1, 0, 2, 3), # 65, 6, 64, 64, not rescaled
                state.permute(1, 0, 2, 3) / self.RESCALER, # 65, 6, 64, 64 rescaled
                sim_id,
            )

        return data

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import torch
    from pathlib import Path

    sys.path.append(os.getcwd())

    @hydra.main(
        version_base=None,
        config_path="../../configurations/dataset", 
        config_name="simulation_smoke.yaml"
    )
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg))
        dataset = SmokeDataset(
            cfg=cfg,
            split="training"
        )
        print("len(dataset): ", len(dataset))

        n_total = 0
        obs_sum = None
        obs_sq_sum = None
        ctrl_sum = None
        ctrl_sq_sum = None

        dataset_dir = Path(dataset.save_dir)
        # Folder names are expected to be sim_{:06d}
        sim_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sim_")])
        n_sims = len(sim_dirs)

        from tqdm import tqdm
        # Initialize variables for sum, squared sum, and count for last-frame smoke value
        last_smoke_sum = 0.0
        last_smoke_sq_sum = 0.0
        last_smoke_count = 0
        print(n_sims)
        # for idx in tqdm(range(36000, n_sims), desc="Processing simulations"):
        for idx in tqdm(range(36000), desc="Processing simulations"):
            data = dataset[idx]
            # data is (states, shape1, shape2, sim_id)
            states = data[0]  # shape [T, 6, 64, 64], already divided by RESCALER

            # observations: density, velocity_x, velocity_y, smoke (= first 4 channels)
            observations = states[:, :4, :, :]   # [T, 4, H, W]
            # control: last two channels
            controls = states[:, 4:, :, :]       # [T, 2, H, W]

            observations = observations.to(torch.float64)
            controls = controls.to(torch.float64)
            # Get last time frame's smoke value (assume smoke is channel 3 of observations)
            last_smoke = observations[-1, 3][0,0]   # shape [64, 64]
            # print("last_smoke: ", last_smoke)
            last_smoke_value = last_smoke.sum().item()
            last_smoke_sum += last_smoke_value
            last_smoke_sq_sum += last_smoke_value ** 2
            last_smoke_count += last_smoke.numel()

            # # observations: [T, 4, 64, 64], controls: [T, 2, 64, 64]
            # # We'll accumulate sum across temporal dimension but keep (C, H, W)
            # if obs_sum is None:
            #     obs_sum = observations.sum(dim=0)          # [4, 64, 64]
            #     obs_sq_sum = (observations ** 2).sum(dim=0)
            #     ctrl_sum = controls.sum(dim=0)              # [2, 64, 64]
            #     ctrl_sq_sum = (controls ** 2).sum(dim=0)
            # else:
            #     obs_sum += observations.sum(dim=0)
            #     obs_sq_sum += (observations ** 2).sum(dim=0)
            #     ctrl_sum += controls.sum(dim=0)
            #     ctrl_sq_sum += (controls ** 2).sum(dim=0)

            # n_total += observations.shape[0]  # number of time steps

        # After the loop, compute the average and standard deviation of the last-frame smoke value across the dataset
        # 0.5848
        # Full 65 time steps:
        # Average last-frame smoke value across all dataset:  0.21777951025911216
        # Std of last-frame smoke value across all dataset:  0.3594170388203331
        # Average last-frame smoke value across 0-35999:  0.22035635935903988
        # Std of last-frame smoke value across 0-35999:  0.36085326418323854
        # Average last-frame smoke value across 36000-39999:  0.19458786835976158
        # Std of last-frame smoke value across 36000-39999: 0.3453588856290529
        avg_last_smoke = last_smoke_sum / last_smoke_count 
        std_last_smoke = (last_smoke_sq_sum / last_smoke_count - avg_last_smoke ** 2) ** 0.5
        print("Average last-frame smoke value across all dataset: ", avg_last_smoke)
        print("Std of last-frame smoke value across all dataset: ", std_last_smoke)

        import pdb; pdb.set_trace()
        # Compute mean and std for observations and controls (per-pixel/channel)
        mean_observation = obs_sum / n_total         # [4, 64, 64]
        std_observation = (obs_sq_sum / n_total - mean_observation ** 2).sqrt()
        mean_control = ctrl_sum / n_total            # [2, 64, 64]
        std_control = (ctrl_sq_sum / n_total - mean_control ** 2).sqrt()

        # Convert to float32 for compatibility & save
        mean_observation = mean_observation.float()
        std_observation = std_observation.float()
        mean_control = mean_control.float()
        std_control = std_control.float()

        np.save("./data/smoke/observation_mean.npy", mean_observation.cpu().numpy())
        np.save("./data/smoke/observation_std.npy", std_observation.cpu().numpy())
        np.save("./data/smoke/control_mean.npy", mean_control.cpu().numpy())
        np.save("./data/smoke/control_std.npy", std_control.cpu().numpy())


        import pdb; pdb.set_trace()

    main()
