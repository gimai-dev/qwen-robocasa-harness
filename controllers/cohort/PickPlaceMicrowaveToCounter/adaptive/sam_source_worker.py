"""CUDA SAM source proposals with natural-color and binary-mask galleries."""
import argparse,json,time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--result',required=True);args=parser.parse_args()
 request=json.loads(Path(args.request).read_text());folder=Path(args.result).parent;root=Path('/home/jli/state/qwen-rgb-sam2')
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
 model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(root/'sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
 result={'model':'sam2.1_hiera_small','device':'cuda','views':{}}
 for view,row in request['views'].items():
  out=folder/(view+'-masks');out.mkdir(exist_ok=True);original=Image.open(row['image_path']).convert('RGB');image=np.array(original);h,w=image.shape[:2];point=row['coarse_point_1000']
  if point is not None:
   point=np.array(point)*[w/1000,h/1000];lower=np.maximum(point-48,0);upper=np.minimum(point+48,[w,h]);xs=np.arange(lower[0]+4,upper[0],8);ys=np.arange(lower[1]+4,upper[1],8);grid=np.array([[x/w,y/h] for y in ys for x in xs]);generator=SAM2AutomaticMaskGenerator(model,points_per_side=None,point_grids=[grid],points_per_batch=16,pred_iou_thresh=.8,stability_score_thresh=.95,crop_n_layers=0);method='coarse ROI ±48px; 8px grid'
  else:
   generator=SAM2AutomaticMaskGenerator(model,points_per_side=32,points_per_batch=16,pred_iou_thresh=.8,stability_score_thresh=.95,crop_n_layers=0);method='full-frame 32x32 grid'
  tic=time.monotonic()
  with torch.inference_mode():anns=generator.generate(image)
  # Food foregrounds can be merged with their gray plate or drain by SAM.
  saturation=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1]
  count,labels,stats,centers=cv2.connectedComponentsWithStats((saturation>25).astype('uint8'),8)
  for label in range(1,count):
   x,y,cw,ch,area=[int(v) for v in stats[label]]
   if not 6<=area<=1500:continue
   if point is not None and np.linalg.norm(centers[label]-point)>48:continue
   anns.append({'segmentation':labels==label,'area':area,'bbox':[x,y,cw,ch],'proposal_method':'pigmented RGB component; semantic source selection still required'})
  elapsed=time.monotonic()-tic;columns=min(4,max(1,len(anns)));size=(columns*256,280*max(1,(len(anns)+columns-1)//columns));natural=Image.new('RGB',size,'white');binary=natural.copy();nd=ImageDraw.Draw(natural);bd=ImageDraw.Draw(binary);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22);candidates=[]
  for i,ann in enumerate(anns):
   mask=ann['segmentation'];yy,xx=np.where(mask);mask_path=out/f'mask-{i}.png';mask_image=Image.fromarray(mask.astype(np.uint8)*255);mask_image.save(mask_path)
   candidate={k:v for k,v in ann.items() if k!='segmentation'};candidate.update(id=i,mask_path=str(mask_path),centroid_pixel=[float(xx.mean()),float(yy.mean())]);candidates.append(candidate)
   box=(max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(w,int(xx.max())+9),min(h,int(yy.max())+9));isolated=Image.composite(original,Image.new('RGB',original.size,(48,48,48)),mask_image);crop=isolated.crop(box);binary_crop=mask_image.convert('RGB').crop(box)
   scale=min(248/crop.width,248/crop.height);render=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)));crop=crop.resize(render);binary_crop=binary_crop.resize(render,Image.Resampling.NEAREST);cx=i%columns*256;cy=i//columns*280;natural.paste(crop,(cx+4,cy+28));binary.paste(binary_crop,(cx+4,cy+28));nd.text((cx+4,cy+4),f'ID {i}',fill='black',font=font);bd.text((cx+4,cy+4),f'ID {i}',fill='black',font=font)
  natural_path,binary_path=out/'natural-crops.png',out/'binary-masks.png';natural.save(natural_path);binary.save(binary_path)
  result['views'][view]={'method':method,'elapsed_seconds':elapsed,'candidates':candidates,'crop_gallery_path':str(natural_path),'binary_gallery_path':str(binary_path)}
  Path(args.result).write_text(json.dumps(result,indent=2))
 result['elapsed_seconds']=time.monotonic()-start;Path(args.result).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
