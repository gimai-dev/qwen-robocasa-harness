"""CUDA SAM2.1 full-frame proposals from a supplied public image."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    folder = Path(args.result).parent
    image = np.array(Image.open(args.image).convert('RGB'))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path('/home/jli/state/qwen-rgb-sam2')
    start = time.monotonic()
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',
                       str(root/'sam2.1_hiera_small.pt'), device='cuda',
                       apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=24, points_per_batch=16,
        pred_iou_thresh=.8, stability_score_thresh=.90, crop_n_layers=0)
    with torch.inference_mode():
        annotations = generator.generate(image)
    rows = []
    board = Image.new('RGB', (1024,280*max(1,(len(annotations)+3)//4)), 'white')
    gallery = Image.new('RGB',board.size,'white')
    draw = ImageDraw.Draw(board)
    crop_draw = ImageDraw.Draw(gallery)
    for index, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        ys,xs = np.where(mask)
        distance = distance_transform_edt(mask)
        y,x = np.unravel_index(distance.argmax(), distance.shape)
        mask_path = folder/f'mask-{index}.png'
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        row = {k:v for k,v in annotation.items() if k!='segmentation'}
        row.update(id=index, mask_path=str(mask_path),
                   max_interior_pixel=[int(x),int(y)], inradius_px=float(distance[y,x]))
        rows.append(row)
        panel = Image.fromarray(np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,axis=2))
        panel.thumbnail((256,256))
        cx,cy = index%4*256,index//4*280
        board.paste(panel,(cx,cy+24))
        draw.text((cx+4,cy+4),f'id={index} q={row["predicted_iou"]:.3f} area={row["area"]}',fill='black')
        crop = Image.fromarray(image).crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
        crop.thumbnail((248,248))
        gallery.paste(crop,(cx+4,cy+24))
        crop_draw.text((cx+4,cy+4),f'id={index}',fill='black')
    board.save(folder/'comparison.png')
    gallery.save(folder/'gallery.png')
    result = {'model':'sam2.1_hiera_small','device':'cuda',
              'method':'full-frame 24x24 prompt grid',
              'elapsed_seconds':time.monotonic()-start,'candidates':rows}
    Path(args.result).write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
