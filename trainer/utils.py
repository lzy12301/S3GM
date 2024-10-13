import os
import torch
import tensorflow as tf
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['epoch'] = loaded_state['epoch']
    state['iteration'] = loaded_state['iteration']
    try:
      state['loss_train'] = loaded_state['loss_train']
      # state['loss_val'] = loaded_state['loss_val']
    except:
      pass
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'epoch': state['epoch'],
    'iteration': state['iteration'],
    'loss_train': state['loss_train'],
    'loss_val': state['loss_val'],
  }
  torch.save(saved_state, ckpt_dir)