import torch

def ckpt_translate(ckpt_path, save_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('diffusion') or key == 'data_mean' or key == 'data_std':
            new_state_dict['model.' + key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    ckpt['state_dict'] = new_state_dict
    torch.save(ckpt, save_path)
