import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import cv2
import datasets
import models
import utils
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from mmcv.runner import load_checkpoint
import datasets
import models
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou




def tensor2PIL(tensor):
    """
    Convert a tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        PIL.Image: PIL image.
    """
    img = transforms.ToPILImage()(tensor)
    return img
# import numpy as np
# from scipy.ndimage import label
# from sklearn.metrics import jaccard_score


import numpy as np
from scipy.ndimage import label
from sklearn.metrics import jaccard_score
import torch
from torch import Tensor

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import jaccard_score
import torch
from torch import Tensor

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import jaccard_score
import torch
from torch import Tensor


def draw_contours(image, mask, color):
    mask = np.squeeze(mask)  # Remove singleton dimensions if present

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # Return the original image if no contours are found

    # Draw the contours on the image
    cv2.drawContours(image, contours, -1, color, 8)

    return image


def convert_tensor_to_image(tensor, data_norm):
    tensor = tensor.cpu().numpy()  # Move tensor to CPU and convert to numpy array
    tensor = np.transpose(tensor, (1, 2, 0))  # Transpose dimensions
    tensor = tensor * data_norm['inp']['div'][0] + data_norm['inp']['sub'][0]  # Apply inverse normalization
    tensor = np.clip(tensor, 0, 1)  # Clip values between 0 and 1
    tensor = tensor * 255  # Scale the values to the 0-255 range
    tensor = tensor.astype(np.uint8)  # Convert to unsigned integer
    return tensor

# def tensor2PIL(tensor):
#     tensor = tensor.cpu()
#     toPIL = transforms.ToPILImage()
#     return np.array(toPIL(tensor)).astype('uint8')



def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_dir=None, runcode=None, dicethre=0.3, datanamelist=None):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0.5], 'div': [0.5]},
            'gt': {'sub': [0.5], 'div': [0.5]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_iou = utils.Averager()
    val_dice = utils.Averager()

    # os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist
     # Create a separate directory for original images
    # orig_save_dir = os.path.join(save_dir, 'original_images')
    # os.makedirs(orig_save_dir, exist_ok=True)
    pred_save_dir = os.path.join(save_dir, runcode)
    os.makedirs(pred_save_dir, exist_ok=True)

    prob_save_dir = os.path.join(save_dir, 'probmap')
    os.makedirs(prob_save_dir, exist_ok=True)
    
    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    for batch in pbar:
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to(device)

            inp = batch['inp']
            pred = torch.sigmoid(model.infer(inp))

            result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
            for p in range(pred.shape[0]):
                imgname = datanamelist[cnt].split('.')[0]

                original_image = convert_tensor_to_image(batch['inp'][p], data_norm)  # Convert to numpy array
                probmap = pred[p].cpu().numpy()
                np.save(os.path.join(prob_save_dir, f'{imgname}.npy'), probmap)

                mask_pred = (pred[p] > 0.5).cpu().numpy().astype(np.uint8)
                mask_gt = batch['gt'][p].cpu().numpy().astype(np.uint8)
            

                iou = calculate_iou(mask_pred, mask_gt)

                val_iou.add(iou, 1)

                # import random
                original_image_with_contours = original_image.copy()

                original_image_with_contours = draw_contours(original_image_with_contours, mask_gt, (255, 255, 0))
            # Save the predicted mask directly as a binary image
                
                pred_mask_path = os.path.join(pred_save_dir, f'predicted_mask_{imgname}.png')
                # cv2.imwrite(pred_mask_path, mask_pred * 255) 
                pred_mask_tensor = torch.tensor(mask_pred * 255, dtype=torch.uint8)
                pred_mask_pil = tensor2PIL(pred_mask_tensor)
                pred_mask_np = np.array(pred_mask_pil)
                plt.imsave(pred_mask_path, pred_mask_np)


 # Multiply by 255 to get pixel values in the range [0, 255]

                original_image_with_contours = draw_contours(original_image_with_contours, mask_pred, (0, 255, 0))

                # Save the modified image
                save_path = os.path.join(save_dir, f'image_{imgname}.png')
                plt.imsave(save_path, original_image_with_contours)
               



        cnt += 1


    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_iou.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--runcode', default='predicted_masks')
    parser.add_argument('--dicethre', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    datanamelist = sorted(os.listdir(spec['dataset']['args']['root_path_1']))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).to(device)
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    save_dir = './contours' + '_' + args.runcode + '/'
    # os.makedirs(pred_save_dir, exist_ok=True)
    metric1, metric2, metric3, metric4, iou = eval_psnr(loader, model,
                                                               data_norm=config.get('data_norm'),
                                                               eval_type=config.get('eval_type'),
                                                               eval_bsize=config.get('eval_bsize'),
                                                               verbose=True,
                                                               save_dir=save_dir,
                                                               runcode=args.runcode,
                                                               dicethre=args.dicethre,
                                                               datanamelist=datanamelist)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    print(iou)