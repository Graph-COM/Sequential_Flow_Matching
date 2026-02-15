import torch
import torch.nn.functional as F

def customized_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in batch.
    
    Args:
        batch: List of tuples, where each tuple is (x_old, x_new, x_gt, t)
               - x_old: [?, ...] - variable length sequence with arbitrary shape
               - x_new: [?, ...] - variable length sequence with same shape as x_old
               - x_gt: [T, ...]  - ground truth sequence in full length
               - conditions: [T, ...] - conditions with same shape as x_gt
               - t: [1] - scalar tensor
    
    Returns:
        Tuple of (x_old_batched, x_new_batched, x_gt_batched, t_batched, pad_masks)
        - x_old_batched: [batch_size, max_len_old_new, ...]
        - x_new_batched: [batch_size, max_len_old_new, ...]
        - x_gt_batched: [batch_size, max_len_gt, ...]
        - conditions_batched: [batch_size, max_len_gt, ...]
        - t_batched: [batch_size, 1]
        - pad_masks: (pad_mask_old_new, pad_mask_gt) - True for non-padded entries
    """
    batch_size = len(batch)
    bundle_size = len(batch[0])
    has_condition = bundle_size == 5
    # Extract all tensors and find maximum lengths
    x_old_list, x_new_list, x_gt_list, t_list = [], [], [], []
    if has_condition:
        condition_list = []
    else:
        condition_list = None
    max_len_old_new = max_len_gt = 0
    
    for b in range(batch_size):
        if has_condition:
            x_old, x_new, x_gt, c, t = batch[b]
            condition_list.append(c)
        else:
            x_old, x_new, x_gt, t = batch[b]
        x_old_list.append(x_old)
        x_new_list.append(x_new)
        x_gt_list.append(x_gt)
        t_list.append(t)
        
        # Update maximum lengths (x_old and x_new have same length)
        max_len_old_new = max(max_len_old_new, x_old.shape[0])
        max_len_gt = max(max_len_gt, x_gt.shape[0])
    
    # Get device, dtype, and shape info from first tensor
    device = x_old_list[0].device
    dtype = x_old_list[0].dtype
    x_old_shape = x_old_list[0].shape[1:]  # Get shape excluding sequence dimension
    x_gt_shape = x_gt_list[0].shape[1:]    # Get shape excluding sequence dimension
    
    # Pad and stack tensors using F.pad for efficiency
    x_old_padded_list, x_new_padded_list, x_gt_padded_list = [], [], []
    
    for b in range(batch_size):
        x_old, x_new, x_gt, t = x_old_list[b], x_new_list[b], x_gt_list[b], t_list[b]
        
        # Pad x_old and x_new to max_len_old_new (they have same length)
        seq_len_old_new = x_old.shape[0]  # x_old and x_new have same length
        if seq_len_old_new < max_len_old_new:
            # Create padding tuple: pad only the first dimension (sequence length)
            # For n-dimensional tensor, we need 2*n padding values (left, right for each dim)
            pad_old_new = [0] * (2 * len(x_old.shape))
            pad_old_new[-1] = max_len_old_new - seq_len_old_new  # Pad at the end of first dim
            pad_old_new = tuple(pad_old_new)
            x_old = F.pad(x_old, pad_old_new, mode='constant', value=0)
            x_new = F.pad(x_new, pad_old_new, mode='constant', value=0)
        x_old_padded_list.append(x_old)
        x_new_padded_list.append(x_new)
        
        # Pad x_gt to max_len_gt
        seq_len_gt = x_gt.shape[0]
        if seq_len_gt < max_len_gt:
            # Create padding tuple for x_gt
            pad_gt = [0] * (2 * len(x_gt.shape))
            pad_gt[-1] = max_len_gt - seq_len_gt  # Pad at the end of first dim
            pad_gt = tuple(pad_gt)
            x_gt = F.pad(x_gt, pad_gt, mode='constant', value=0)
        x_gt_padded_list.append(x_gt)
    
    # Stack padded tensors along batch dimension (batch_size first)
    x_old_batched = torch.stack(x_old_padded_list, dim=0)  # [batch_size, max_len_old_new, ...]
    x_new_batched = torch.stack(x_new_padded_list, dim=0)  # [batch_size, max_len_old_new, ...]
    x_gt_batched = torch.stack(x_gt_padded_list, dim=0)    # [batch_size, max_len_gt, ...]
    if has_condition:
        conditions_batched = torch.stack(condition_list, dim=0) # [batch_size, max_len_gt, ...]
    else:
        conditions_batched = None
    t_batched = torch.stack(t_list, dim=0)                 # [batch_size, 1]
    
    # Create padding masks efficiently (only need one mask for x_old/x_new)
    pad_mask_old_new = torch.zeros(batch_size, max_len_old_new, dtype=torch.bool, device=device)
    pad_mask_gt = torch.zeros(batch_size, max_len_gt, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        seq_len_old_new = x_old_list[b].shape[0]  # x_old and x_new have same length
        seq_len_gt = x_gt_list[b].shape[0]
        
        pad_mask_old_new[b, :seq_len_old_new] = True
        pad_mask_gt[b, :seq_len_gt] = True
    
    return x_old_batched, x_new_batched, x_gt_batched, conditions_batched, t_batched, (pad_mask_old_new, pad_mask_gt)