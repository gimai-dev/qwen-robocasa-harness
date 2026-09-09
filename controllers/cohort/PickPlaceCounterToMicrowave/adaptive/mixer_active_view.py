"""Aim the public wrist camera before committing to an unlocalized placement."""
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .image_servo import _rotation_xyzw
from .pose_skill import _pose_arrays,plan_pose_segment
from .sam_circle_destination import _fit_circle


def approximate_floor_for_viewing(unresolved,views,state):
    """A weak circle pair may guide a view, but cannot authorize placement."""
    evidence=unresolved['evidence']
    first=evidence.get('initial_selection',{}).get('refinement',{}).get('convex_mask_path')
    if first is None:return None
    for attempt in evidence.get('history_attempts',[]):
        other=attempt.get('selection',{}).get('refinement',{}).get('convex_mask_path')
        if other is None:continue
        index=attempt['view_index']
        masks=[cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)>0 for path in [first,other]]
        try:
            fit=_fit_circle(masks,[views[0]['calibration'],views[index]['calibration']],state)
        except ValueError:continue
        return {'status':'view_hint','target_world_m':fit['target_world_m'],
                'view_indices':[0,index],'purpose':'camera aiming only; placement still requires supported geometry'}
    return None


def camera_view_plan(q,state,camera,target_world,advance_m=.35):
    """Turn a clear, unloaded wrist toward the hint, accounting for its offset."""
    position,rotation=_pose_arrays(q)
    base_rotation=np.asarray(_rotation_xyzw(state['state.base_rotation']))
    base=np.asarray(state['state.base_position'])
    eef_world=base+base_rotation@position
    eef_rotation=base_rotation@rotation
    relative_rotation=eef_rotation.T@np.asarray(camera['camera_xmat_world'])
    camera_offset=eef_rotation.T@(np.asarray(camera['camera_position_world_m'])-eef_world)
    target=np.asarray(target_world)
    toward=base_rotation.T@(target-base)-position
    toward[2]=0.
    toward/=np.linalg.norm(toward)
    view_position=position+advance_m*toward
    view_world=base+base_rotation@view_position
    for roll in [0.,math.pi/2,-math.pi/2,math.pi]:
        trial=eef_rotation.copy()
        for _ in range(6):
            origin=view_world+trial@camera_offset
            z=origin-target;z/=np.linalg.norm(z)
            x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x)
            y=np.cross(z,x)
            desired_camera=np.column_stack([x,y,z])@Rotation.from_euler('z',roll).as_matrix()
            trial=desired_camera@relative_rotation.T
        desired=base_rotation.T@trial
        path=plan_pose_segment(q,view_position,desired)
        if path['status']!='kinematically_reachable':continue
        # Prefer the upright image. The shorter rolled view hid the floor and
        # Qwen selected the round motor housing instead of the receiving bowl.
        delta=origin-target
        elevation=math.degrees(math.atan2(delta[2],np.linalg.norm(delta[:2])))
        return {'status':'planned','position_base_m':view_position.tolist(),
                'rotation_base':desired.tolist(),'restore_position_base_m':position.tolist(),
                'restore_rotation_base':rotation.tolist(),'camera_roll_rad':roll,
                'camera_elevation_deg':elevation,'path':path,'hint_world_m':target.tolist()}
    return {'status':'unresolved','reason':'camera_view_pose_unreachable'}
