import apache_beam   # Needs to be imported separately to avoid TypingError
try:
    import weatherbench2
except ImportError:
    print("weatherbench2 not found. This is fine if you are not downloading the dataset.")
import xarray as xr
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from omegaconf import DictConfig
import json
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

class WeatherDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.n_frames = cfg.n_frames * cfg.frame_skip
        self.frame_skip = cfg.frame_skip
        self.start_year = 1959
        self.end_year = 2023
        if split == "training":
            self.file_years = [year for year in range(1959, 2011)]
        elif split == 'finetune':
            self.file_years = [year for year in range(2011, 2021)]
            #self.file_years = [year for year in range(2011, 2013)]
        else:
            self.file_years = [year for year in range(2021, 2023)]

        self.save_dir = Path(cfg.save_dir)
        if not (self.save_dir/'metadata.json').exists(): # check if dataset is saved
            print('data path not found, start to download data...')
            self.save_dir.mkdir(exist_ok=True, parents=True)
            self._download()
        self.clips_per_file = self._compute_clips_per_file()
        self.cum_clips_per_file = np.cumsum(self.clips_per_file)

        # TODO: this is hard-coded
        if split == 'finetune':
            n_finetune = 5000
            #n_finetune = 10
            idx = np.searchsorted(self.cum_clips_per_file, n_finetune)
            if idx > 0:
                left = n_finetune - self.cum_clips_per_file[idx - 1]
                self.clips_per_file = np.concatenate((self.clips_per_file[:idx], [left]))
                self.cum_clips_per_file = np.cumsum(self.clips_per_file)
            else:
                self.clips_per_file = np.array([n_finetune])
                self.cum_clips_per_file = np.cumsum(self.clips_per_file)
        # shuffle clips for more diverse evaluation
        random.seed(0)
        self.idx_remap = list(range(self.__len__()))
        random.shuffle(self.idx_remap)

    def _compute_clips_per_file(self):
        with open(self.save_dir/'metadata.json', 'r') as f:
            # Use json.load to deserialize the JSON content into a Python object
            len_per_file = json.load(f)
        len_per_file = [len_per_file[i-self.start_year] for i in self.file_years]
        return np.array(len_per_file) - self.n_frames + 1

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        #year = str(self.start_year + file_idx)
        year = str(self.file_years[file_idx])
        file_path = self.save_dir / 'year{}.npy'.format(year)
        data = np.load(file_path)
        data = data[frame_idx : frame_idx + self.n_frames]
        assert len(data) == self.n_frames
        return data[:: self.frame_skip]

    def split_idx(self, idx):
        file_idx = np.argmax(self.cum_clips_per_file > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_file, (1, 0))[file_idx]
        return file_idx, frame_idx

    def _download(self):
        # lazy load data from weatherbench2
        print('lazy loading a massive dataset from weatherbench, please wait...')
        full_dataset = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
        print('Full dataset is lazy loaded.')
        # single-level (2D) variables
        surface_vars = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure'
        ]
        # pressure-level (3D) variables
        pressure_vars = [
            'geopotential',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind'
        ]
        # The common levels you want to select for 3D variables
        levels = [500, 850, 1000]
        # select a 32 * 32 grid
        lat_slice = slice(45.0, 37.25)  # N -> S (45.0 - 7.75 = 37.25)
        lon_slice = slice(115.0, 122.75)  # W -> E (115.0 + 7.75 = 122.75)
        print('Slice a subset from full dataset...')
        dataset = full_dataset[surface_vars+pressure_vars].sel(
            latitude=lat_slice,
            longitude=lon_slice,
            # This selects the target levels ONLY for the variables that have a 'level' dimension
            level=levels,
        )

        print('Start to download the whole dataset chunked per year...')
        from tqdm import tqdm
        pbar = tqdm(total=self.end_year-self.start_year+1, initial=self.start_year,desc="Downloading by years")
        len_per_file = []
        for year in range(self.start_year, self.end_year):
            if (self.save_dir/f'year{year}.npy').exists():
                data_year = np.load(self.save_dir / 'metadata.json')
            else:
                year = str(year)
                start_date = year + '-01-01'
                end_date = year + '-12-31'
                dataset_year = dataset.sel(time=slice(start_date, end_date))
                # List to hold all channel data
                channels = []
                # Surface variables (2D): need to expand to add channel dimension
                for var_name in surface_vars:
                    var_data = dataset_year[var_name].values  # shape: [time, latitude, longitude]
                    # Expand dimension to [time, 1, latitude, longitude]
                    var_data = np.expand_dims(var_data, axis=1)
                    channels.append(var_data)
                for var_name in pressure_vars:
                    var_data = dataset_year[var_name].values  # shape: [time, level, latitude, longitude]
                    # Each level becomes a separate channel
                    # var_data is already [time, level, latitude, longitude] where level=3
                    # We want to treat each level as a separate channel, so this is already correct
                    channels.append(var_data)

                # Concatenate all channels: [time, total_channels, latitude, longitude]
                # Surface vars: 4 channels (each has 1 channel)
                # Pressure vars: 4 * 3 = 12 channels (each has 3 levels)
                # Total: 16 channels
                data_year = np.concatenate(channels, axis=1)  # shape: [time, 16, latitude, longitude]
                # save data to disk
                np.save(self.save_dir / f'year{year}.npy', data_year)
            len_per_file.append(len(data_year))
            pbar.update(1)
        import json
        with open(self.save_dir / 'metadata.json', 'w') as f:
            json.dump(len_per_file, f)

    def __len__(self):
        return self.clips_per_file.sum()

    def get(self, idx):
        return self.__getitem__(idx)
        
def get_burgers_preprocess(
    rescaler=None, 
    stack_u_and_f=False, 
    pad_for_2d_conv=False, 
    partially_observed_fill_zero_unobserved=None, 
):
    if rescaler is None:
        raise NotImplementedError('Should specify rescaler. If no rescaler is not used, specify 1.')
    
    def preprocess(db):
        '''We are only returning f and u for now, in the shape of 
        (u0, u1, ..., f0, f1, ...)
        '''
        
        u = db['u']
        f = db['f']
        f = f[:,:15]

        fill_zero_unobserved = partially_observed_fill_zero_unobserved
        if fill_zero_unobserved is not None:
            if fill_zero_unobserved == 'front_rear_quarter':
                u = u.squeeze()
                nx = u.shape[-1]
                u[..., nx // 4: (nx * 3) // 4] = 0
            else:
                raise ValueError('Unknown partially observed mode')

        if stack_u_and_f:
            assert pad_for_2d_conv
            nt = f.size(-2)
            f = nn.functional.pad(f, (0, 0, 0, 16 - nt), 'constant', 0)
            u = nn.functional.pad(u, (0, 0, 0, 15 - nt), 'constant', 0)
            u_target = u  
            data = torch.stack((u, f, u_target), dim=1) 
        else:
            assert not pad_for_2d_conv
            data = torch.cat((u, f), dim=1)
    
        data = data / rescaler
        return data

    return preprocess



class LegacyBurgersDataset(Dataset):
    def __init__(
        self, 
        fname, 
        preprocess=get_burgers_preprocess('all'),  
        load_all=True
    ):
        '''
        Arguments:

        '''
        self.load_all = load_all
        if load_all:
            self.db = torch.load(fname)
            self.x = preprocess(self.db)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.load_all:
            return self.x.size(0)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.load_all:
            return self.x[idx]
        else:
            raise NotImplementedError

    def get(self, idx):
        return self.__getitem__(idx)
    
    def len(self):
        return self.__len__()


if __name__=='__main__':

    tmp = LegacyBurgersDataset('data/pde/train_data', 
        preprocess=get_burgers_preprocess(
            rescaler=10, 
            stack_u_and_f=True, 
            pad_for_2d_conv=True, 
            partially_observed_fill_zero_unobserved = None 
        ) 
    )

    import pdb; pdb.set_trace()
