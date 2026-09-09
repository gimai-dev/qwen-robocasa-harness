"""Conservative grasp width from intersecting calibrated RGB silhouettes.

The volume is an outer shape approximation, not a recovered mesh or support
height. Its horizontal width can bound a grasp even when a vertical silhouette
cannot be flattened onto the stereo-center plane.
"""
from itertools import combinations
import cv2
import numpy as np
from .rgb_grounding import _triangulate
from .image_servo import _rotation_xyzw

VOXEL_STEP_M=.003
SEARCH_RADIUS_M=.15

def _project(world,camera):
 local=(world-np.asarray(camera['camera_position_world_m']))@np.asarray(camera['camera_xmat_world'])
 depth=-local[:,2]
 return (np.rint(camera['cx_px']+camera['fx_px']*local[:,0]/depth).astype(int),
         np.rint(camera['cy_px']-camera['fy_px']*local[:,1]/depth).astype(int),depth)

def ground_source_hull(views,calibration,state):
 rotation=np.asarray(_rotation_xyzw(state['state.base_rotation']))
 base=np.asarray(state['state.base_position'])
 candidates=[];rejected=[]
 for a,b in combinations(views,2):
  pixels={v:views[v]['pixel'] for v in (a,b)}
  stereo=_triangulate(pixels,calibration)
  if stereo is None:continue
  center,gap,sensitivity=stereo
  axis=np.arange(-SEARCH_RADIUS_M,SEARCH_RADIUS_M+VOXEL_STEP_M/2,VOXEL_STEP_M)
  world=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)+center
  for view in (a,b):
   mask=views[view]['mask'];u,v,depth=_project(world,calibration[view])
   inside=(u>=0)&(u<mask.shape[1])&(v>=0)&(v<mask.shape[0])&(depth>0)
   indices=np.flatnonzero(inside);keep=np.zeros(len(world),bool)
   keep[indices]=mask[v[indices],u[indices]];world=world[keep]
  if len(world)<6:
   rejected.append({'views':[a,b],'reason':'no_common_silhouette_volume'});continue
  if np.max(np.abs(world-center))>=SEARCH_RADIUS_M-VOXEL_STEP_M/2:
   rejected.append({'views':[a,b],'reason':'silhouette_volume_exceeds_search_region'});continue
  coverage={}
  for view in (a,b):
   mask=views[view]['mask'];u,v,_=_project(world,calibration[view]);render=np.zeros_like(mask);render[v,u]=True
   coverage[view]=float((render&mask).sum()/mask.sum())
  if min(coverage.values())<.95:
   rejected.append({'views':[a,b],'reason':'incompatible_or_underresolved_silhouettes','coverage':coverage});continue
  points=(world-base)@rotation
  box=cv2.boxPoints(cv2.minAreaRect(points[:,:2].astype(np.float32)))
  edges=np.roll(box,-1,axis=0)-box;closing=edges[np.argmin(np.linalg.norm(edges,axis=1))]
  closing/=np.linalg.norm(closing)
  # Account for the full projected voxel square, not just occupied centers.
  width=float(np.ptp(points[:,:2]@closing)+VOXEL_STEP_M*np.abs(rotation@np.r_[closing,0.]).sum())
  if width>.08:
   rejected.append({'views':[a,b],'reason':'hull_width_exceeds_gripper','width_m':width});continue
  yaw=float((np.arctan2(closing[1],closing[0])+np.pi/2)%np.pi-np.pi/2)
  closing=np.array([np.cos(yaw),np.sin(yaw),0.])
  stereo_base=rotation.T@(center-base)
  bounds=np.array([np.min(points@closing),np.max(points@closing)])
  # Equalize the two open-finger clearances across the accepted outer hull.
  # The stereo height and coordinate along the object remain unchanged.
  grasp=stereo_base+closing*(bounds.mean()-stereo_base@closing)
  candidates.append({'feature':'object_center','grasp_candidate':True,'paired_points':pixels,
   'target_world_m':(base+rotation@grasp).tolist(),'target_base_m':grasp.tolist(),
   'object_center_base_m':stereo_base.tolist(),'object_center_world_m':center.tolist(),
   'hull_closing_bounds_base_m':bounds.tolist(),
   'hull_height_bounds_base_m':[float(points[:,2].min()),float(points[:,2].max())],
   'ray_gap_m':gap,'one_pixel_sensitivity_m':sensitivity,'projected_width_m':width,
   'jaw_closing_axis_base':[float(np.cos(yaw)),float(np.sin(yaw)),0.],
   'jaw_closing_yaw_rad':yaw,'geometry_method':'calibrated_silhouette_intersection',
   'voxel_step_m':VOXEL_STEP_M,'silhouette_coverage':coverage,'hull_voxels':len(world)})
 return candidates,rejected
