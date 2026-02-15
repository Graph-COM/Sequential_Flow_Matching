import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from omegaconf import DictConfig
import random
import sys, os
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


class FinetuningSmokeDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training", alg_cfg: DictConfig = None):
        # Store config and split for later use
        self.cfg = cfg
        self.alg_cfg = alg_cfg
        self.split = split
        self.update_strategy = alg_cfg.update_strategy
        self.last_frame_update = alg_cfg.last_frame_update
        self.pretrained_model = 'df' if 'diffusion' in alg_cfg else 'flow'

        # Load the inference results BEFORE calling super().__init__()
        self.inference_data = self._load_inference_results()

        # Extract the data
        self.ground_truth = self.inference_data['ground_truth']  # Ground truth
        #self.conditions = self.inference_data['conditions']
        self.model_predictions = self.inference_data['model_pred']  # Model predictions
        self.pred_time = self.inference_data['pred_time']

        # Get metadata from the inference results
        self.frame_stack = self.inference_data.get('frame_stack', 1)
        self.open_loop_horizon = self.inference_data.get('open_loop_horizon', 1)
        self.context_frames = self.inference_data.get('context_frames', 1)

        # Calculate clips per trajectory and create index mapping
        self._calculate_clips_per_trajectory()
        self._create_index_mapping()

        super().__init__()

    def _load_inference_results(self) -> dict:
        """
        Load the inference results from the .pt file.
        """
        # Construct the path to the inference results
        # We need to construct the path manually since self.save_dir isn't set yet
        save_dir = Path(self.cfg.save_dir)
        algorithm_name = self.alg_cfg._name + '_' + self.update_strategy
        inference_path = save_dir / 'fine-tuning' / ("%s.pt" % (algorithm_name))

        if not inference_path.exists():
            raise FileNotFoundError(
                f"Inference results not found at {inference_path}. "
                "Please run inference first to generate the fine-tuning dataset."
            )

        print(f"Loading inference results from {inference_path}")
        inference_data = torch.load(inference_path, map_location='cpu')

        print(f"Loaded inference data with keys: {list(inference_data.keys())}")
        print(f"Ground truth shape: {inference_data['ground_truth'].shape}")
        #print(f"Model predictions shape: {inference_data['model_pred'].shape}")

        if self.alg_cfg.get('n_fine_tuning_data') is not None:
            print("Using %d out of total %d finetuning data" % (self.alg_cfg.get('n_fine_tuning_data'),
                                                          inference_data['ground_truth'].shape[1]))
            num_ft = self.alg_cfg.get('n_fine_tuning_data')
            inference_data['ground_truth'] = inference_data['ground_truth'][:, :num_ft]
            #inference_data['conditions'] = inference_data['conditions'][:, :num_ft]
            for i in range(len(inference_data['model_pred'])):
                inference_data['model_pred'][i] = inference_data['model_pred'][i][:, :num_ft]
                inference_data['pred_time'][i] = inference_data['pred_time'][i][:num_ft]
            print(f"Final Ground truth shape: {inference_data['ground_truth'].shape}")

        return inference_data

    def _calculate_clips_per_trajectory(self):
        """
        Calculate how many samples can be extracted from the data.
        With the new structure: ground_truth (T*B*6*H*W) and model_predictions (H*B*W*6*H*W)
        We have B*(W-1) total samples where:
        - B is the number of trajectories
        - W is the number of prediction attempts in model_predictions
        - We use W-1 because we need consecutive pairs (j and j+1)
        """
        # Get the dimensions
        num_trajectories = self.ground_truth.shape[1]  # B dimension
        num_prediction_attempts = len(self.model_predictions)  # W dimension

        # We can create samples for each trajectory and each consecutive pair of prediction attempts
        # Total samples = B * (W-1)
        samples_per_trajectory = num_prediction_attempts - 1

        self.clips_per_trajectory = np.array([samples_per_trajectory] * num_trajectories, dtype=np.int32)
        self.cum_clips_per_trajectory = np.cumsum(self.clips_per_trajectory)

        print(f"Generated {samples_per_trajectory} samples per trajectory from {num_trajectories} trajectories")
        print(f"Total samples: {self.clips_per_trajectory.sum()}")
        print(f"Ground_truth shape: {self.ground_truth.shape}")
        #print(f"Model predictions shape: {self.model_predictions.shape}")

    def _create_index_mapping(self):
        """
        Create index mapping for random access to clips.
        """
        total_clips = self.clips_per_trajectory.sum()
        self.idx_remap = list(range(total_clips))
        random.seed(0)  # For reproducible shuffling
        random.shuffle(self.idx_remap)

    def get_data_paths(self, split):
        """
        For fine-tuning dataset, we don't have traditional file paths.
        Return a dummy path list.
        """
        # This method is called during base class initialization, so we need to handle
        # the case where inference data might not be loaded yet
        if hasattr(self, 'ground_truth'):
            return [Path("dummy")] * self.ground_truth.shape[1]
        else:
            # Fallback: return a dummy path that will be overridden later
            return [Path("dummy")]

    def get_data_lengths(self, split):
        """
        Return the length of each trajectory in the inference results.
        """
        # This method is called during base class initialization, so we need to handle
        # the case where inference data might not be loaded yet
        if hasattr(self, 'ground_truth'):
            trajectory_length = self.ground_truth.shape[0]
            return [trajectory_length] * self.ground_truth.shape[1]
        else:
            # Fallback: return a dummy length that will be overridden later
            return [100]

    def split_idx(self, idx):
        """
        Split the global index into trajectory index and prediction attempt index.
        """
        trajectory_idx = np.argmax(self.cum_clips_per_trajectory > idx)
        prediction_idx = idx - np.pad(self.cum_clips_per_trajectory, (1, 0))[trajectory_idx]
        return trajectory_idx, prediction_idx

    def __len__(self):
        """
        Return the total number of clips available.
        """
        return self.clips_per_trajectory.sum()

    def __getitem__(self, idx):
        """
        Get a training sample for fine-tuning.

        Returns:
            Tuple of (old_prediction_with_context, updated_prediction_with_context, ground_truth, conditions, physical_time)
            where:
            - old_prediction_with_context: [context + old_prediction] - shape (context_len + pred_len, 6, H, W)
            - updated_prediction_with_context: [context + updated_prediction] - shape (context_len + pred_len, 6, H, W)
            - ground_truth: [full_trajectory] - shape (T, 6, H, W)
            - conditions: [full_trajectory] - shape (T, 2, H, W) or (T, 6, H, W) depending on structure
            - physical_time: [1] - scalar tensor indicating prediction time step
        """
        # Apply index remapping for shuffling
        idx = self.idx_remap[idx]

        # Split index into trajectory and prediction attempt indices
        trajectory_idx, prediction_idx = self.split_idx(idx)

        # Extract the data according to the structure:
        # ground_truth: T*B*6*H*W
        # model_predictions: List of (H*B*6*H*W)

        # (1) old_prediction: y[1:, i, j] - shape (H-1)*6*H*W
        old_prediction = self.model_predictions[prediction_idx][1:, trajectory_idx]

        # (2) updated_prediction: y[:-1, i, j+1] - shape (H-1)*6*H*W
        if self.update_strategy == 'fixed_horizon':
            # for fixed horizon update, we need to decide how to handle the last frame prediction, as we don't have that
            # estimation before
            if self.last_frame_update == 'from_prev_frame':
                old_prediction = torch.cat([old_prediction, old_prediction[-1:]], dim=0)
            elif self.last_frame_update == 'from_noise':
                old_prediction = torch.cat([old_prediction, torch.zeros_like(old_prediction[-1:])], dim=0)
            updated_prediction = self.model_predictions[prediction_idx+1][:, trajectory_idx]
        elif self.update_strategy == 'shrinking_horizon':
            updated_prediction = self.model_predictions[prediction_idx+1][:, trajectory_idx]

        # smoke control is fixed horizon at beginning and shrinking horizon near the end, so we need to pad zeros
        # in case updated_prediction is shorter than old_prediction
        assert old_prediction.shape[0] >= updated_prediction.shape[0]
        if old_prediction.shape[0] > updated_prediction.shape[0]:
            #pad = [0, 0] * (updated_prediction.dim() - 1) + [0, old_prediction.shape[0] - updated_prediction.shape[0]]
            #updated_prediction = F.pad(updated_prediction, pad)
            old_prediction = old_prediction[:updated_prediction.shape[0]] # simply dropped the padded entries
        assert old_prediction.shape == updated_prediction.shape
        # (3) ground_truth: x[:, i] - shape T*6*H*W
        ground_truth = self.ground_truth[:, trajectory_idx].to(dtype=old_prediction.dtype)

        # (4) conditions (i.e., control)
        #conditions = self.conditions[:, trajectory_idx]

        # (5) context_frame_index: j+1+self.context_frames/self.frame_stack - shape 1
        physical_time = self.pred_time[prediction_idx+1][trajectory_idx]
        physical_time = torch.tensor([physical_time])

        # (6) concatenate context frames to old and updated prediction
        # For smoke, we use one-frame context at physical_time - 1
        context = ground_truth[physical_time-1:physical_time]
        old_prediction_with_context = torch.cat([context, old_prediction], dim=0)
        updated_prediction_with_context = torch.cat([context, updated_prediction], dim=0)

        return old_prediction_with_context, updated_prediction_with_context, ground_truth, physical_time

