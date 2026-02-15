import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from omegaconf import DictConfig
import random

class FinetuningMaze2dDataset(Dataset):
    """
    Dataset class for fine-tuning maze planning that loads data from saved inference results.
    
    This dataset loads the saved inference results from the pretrained model,
    which contains:
    - ground_truth: Ground truth trajectories (T, total_samples, bundle_dim)
    - model_pred: Model predictions list [(T_plan, total_samples, bundle_dim), ...]
    - pred_time: Prediction timesteps list [(total_samples,), ...]
    - start: Start states (total_samples, bundle_dim)
    - goal: Goal states (total_samples, bundle_dim)
    """
    
    def __init__(self, cfg: DictConfig, split: str = "training", alg_cfg: DictConfig = None):
        # Store config and split for later use
        self.cfg = cfg
        self.alg_cfg = alg_cfg
        self.split = split
        self.update_strategy = alg_cfg.update_strategy
        self.last_frame_update = alg_cfg.last_frame_update
        
        # Load the inference results
        self.inference_data = self._load_inference_results()
        
        # Extract the data
        self.goals = self.inference_data['goals']
        self.ground_truth = self.inference_data['ground_truth']  # (T, total_samples, bundle_dim)
        self.model_predictions = self.inference_data['model_pred']  # List of (T_plan, total_samples, bundle_dim)
        self.pred_time = self.inference_data['pred_time']  # List of (total_samples,)
        
        # Get metadata from the inference results
        self.frame_stack = self.inference_data.get('frame_stack', 10)
        self.open_loop_horizon = self.inference_data.get('open_loop_horizon', 50)
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
        save_dir = Path(self.cfg.save_dir)
        
        # Use algorithm_name from config if specified, otherwise auto-generate
        if self.alg_cfg.get('dataset_name') is not None:
            algorithm_name = self.alg_cfg.dataset_name
        else:
            # Extract maze size from env_id or dataset name
            maze_size = None
            if hasattr(self.cfg, 'env_id'):
                if 'medium' in self.cfg.env_id:
                    maze_size = 'medium'
                elif 'large' in self.cfg.env_id:
                    maze_size = 'large'
            elif hasattr(self.cfg, '_name'):
                if 'medium' in self.cfg._name:
                    maze_size = 'medium'
                elif 'large' in self.cfg._name:
                    maze_size = 'large'
            
            # Build algorithm name with maze size if available
            if maze_size:
                algorithm_name = self.alg_cfg._name + '_' + self.update_strategy + '_' + maze_size
            else:
                algorithm_name = self.alg_cfg._name + '_' + self.update_strategy
        
        inference_path = save_dir / 'fine-tuning' / ("%s.pt" % algorithm_name)
        
        if not inference_path.exists():
            raise FileNotFoundError(
                f"Inference results not found at {inference_path}. "
                "Please run inference first to generate the fine-tuning dataset."
            )
        
        print(f"Loading inference results from {inference_path}")
        inference_data = torch.load(inference_path, map_location='cpu')
        
        print(f"Loaded inference data with keys: {list(inference_data.keys())}")
        print(f"Ground truth shape: {inference_data['ground_truth'].shape}")
        print(f"Number of prediction windows: {len(inference_data['model_pred'])}")
        
        if self.alg_cfg.get('n_fine_tuning_data') is not None:
            num_ft = self.alg_cfg.get('n_fine_tuning_data')
            print(f"Using {num_ft} out of total {inference_data['ground_truth'].shape[1]} finetuning data")
            inference_data['ground_truth'] = inference_data['ground_truth'][:, :num_ft]
            for i in range(len(inference_data['model_pred'])):
                inference_data['model_pred'][i] = inference_data['model_pred'][i][:, :num_ft]
                inference_data['pred_time'][i] = inference_data['pred_time'][i][:num_ft]
            print(f"Final Ground truth shape: {inference_data['ground_truth'].shape}")
        
        return inference_data
    
    def _calculate_clips_per_trajectory(self):
        """
        Calculate how many samples can be extracted from the data.
        We have num_trajectories * (num_prediction_windows - 1) total samples
        because we need consecutive pairs (j and j+1) of prediction windows.
        """
        num_trajectories = self.ground_truth.shape[1]  # total_samples dimension
        num_prediction_windows = len(self.model_predictions)  # number of planning steps
        
        # We can create samples for each trajectory and each consecutive pair of prediction attempts
        # Total samples = num_trajectories * (num_prediction_windows - 1)
        samples_per_trajectory = num_prediction_windows - 1
        
        self.clips_per_trajectory = np.array([samples_per_trajectory] * num_trajectories, dtype=np.int32)
        self.cum_clips_per_trajectory = np.cumsum(self.clips_per_trajectory)
        
        print(f"Generated {samples_per_trajectory} samples per trajectory from {num_trajectories} trajectories")
        print(f"Total samples: {self.clips_per_trajectory.sum()}")
    
    def _create_index_mapping(self):
        """
        Create index mapping for random access to clips.
        """
        total_clips = self.clips_per_trajectory.sum()
        self.idx_remap = list(range(total_clips))
        random.seed(0)  # For reproducible shuffling
        random.shuffle(self.idx_remap)
    
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
            Tuple of (old_prediction_with_context, updated_prediction_with_context, ground_truth, pred_time)
            where:
            - old_prediction_with_context: [context + old_prediction] - shape (context_len + pred_len, bundle_dim)
            - updated_prediction_with_context: [context + updated_prediction] - shape (context_len + pred_len, bundle_dim)
            - ground_truth: [full_trajectory] - shape (T, bundle_dim)
            - pred_time: [1] - scalar tensor indicating prediction time step
        """
        # Apply index remapping for shuffling
        idx = self.idx_remap[idx]
        
        # Split index into trajectory and prediction attempt indices
        trajectory_idx, prediction_idx = self.split_idx(idx)
        
        # Extract old prediction (from prediction_idx)
        old_prediction = self.model_predictions[prediction_idx][:, trajectory_idx]  # (T_plan, bundle_dim)
        
        # Get prediction time step
        physical_time = self.pred_time[prediction_idx + 1][trajectory_idx]
        physical_time = torch.tensor([physical_time])

        # Extract ground truth trajectory
        ground_truth = self.ground_truth[:, trajectory_idx]  # (T, bundle_dim)
        
        # Extract updated prediction (from prediction_idx + 1)
        # Note that shrinking_horizon is the standard update strategy for maze planning.
        if self.update_strategy == 'fixed_horizon':
            if self.last_frame_update == 'from_prev_frame':
                # Use last frame from old prediction
                old_prediction = torch.cat([old_prediction, old_prediction[-1:]], dim=0)
            elif self.last_frame_update == 'from_noise':
                # Use zero/noise for last frame
                old_prediction = torch.cat([old_prediction, torch.zeros_like(old_prediction[-1:])], dim=0)
            updated_prediction = self.model_predictions[prediction_idx + 1][:, trajectory_idx]  # (T_plan, bundle_dim)
        elif self.update_strategy == 'shrinking_horizon':
            # we ditch the first open loop prediction in old prediction and replace it with gt
            old_prediction = old_prediction[self.open_loop_horizon:]
            updated_prediction = self.model_predictions[prediction_idx + 1][:, trajectory_idx]

        assert old_prediction.shape == updated_prediction.shape
        
        # Concatenate context frames to old and updated prediction
        # For maze planning, context is the last frame at time = physical_time - 1
        context = ground_truth[physical_time-1:physical_time]  # (pred_time, bundle_dim)
        #context = context.tile([self.frame_stack, 1])
        old_prediction_with_context = torch.cat([context, old_prediction], dim=0)  # (pred_time + T_plan, bundle_dim)
        updated_prediction_with_context = torch.cat([context, updated_prediction], dim=0)  # (pred_time + T_plan, bundle_dim)

        goals = self.goals[trajectory_idx]

        return old_prediction_with_context, updated_prediction_with_context, ground_truth, goals, physical_time

