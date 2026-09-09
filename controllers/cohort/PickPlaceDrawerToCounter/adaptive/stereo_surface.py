"""Surface geometry from simultaneous wide-baseline RGB cameras.

A semantic current-frame floor mask selects the surface. Previous-frame features
are unconstrained by a manual ROI. Static correspondence, planar reprojection,
and camera translation must support the output before it becomes a target.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .image_servo import _camera_geometry, _closest_ray_points, _pixel_ray_world, _rotation_xyzw


def _decode(data):
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


def _triangulate(a,b,geometry):
    oa,da=_pixel_ray_world(tuple(a),geometry[0])
    ob,db=_pixel_ray_world(tuple(b),geometry[1])
    try:
        pa,pb,gap,sa,sb=_closest_ray_points(oa,da,ob,db)
    except ValueError:
        return None
    if sa<=0 or sb<=0:
        return None
    return (np.asarray(pa)+pb)/2,float(gap),(oa,da)


def _project(point,calibration):
    local=np.asarray(calibration['camera_xmat_world']).T @ (
        point-np.asarray(calibration['camera_position_world_m']))
    return np.array([calibration['fx_px']*local[0]/-local[2]+calibration['cx_px'],
                     -calibration['fy_px']*local[1]/-local[2]+calibration['cy_px']])


def stereo_surface_point(previous_image_bytes,current_image_bytes,previous_calibration,
                         current_calibration,current_mask,public_state):
    """Return a serializable grounded point or unresolved result, with evidence.

    ``current_mask`` is semantic mask PNG bytes or a 2D array, supplied by the
    caller's perception tool. No simulator object pose, depth, or default camera
    extrinsics are used. A drawer floor is expected to be approximately level.
    """
    evidence={'method':'SIFT ratio; calibrated static rays; RANSAC homography; unique matches; calibrated plane reprojection',
              'selection':'current semantic mask; full previous RGB','matches':[]}
    def unresolved(reason):
        return {'status':'unresolved','target_base_m':None,'target_world_m':None,
                'evidence':{**evidence,'reason':reason}}
    gray=[_decode(previous_image_bytes),_decode(current_image_bytes)]
    mask=_decode(current_mask) if isinstance(current_mask,(bytes,bytearray)) else np.asarray(current_mask)
    mask=((mask>0)*255).astype(np.uint8)
    baseline=float(np.linalg.norm(np.asarray(previous_calibration['camera_position_world_m'])-
                                  current_calibration['camera_position_world_m']))
    evidence['camera_baseline_m']=baseline
    if baseline<1e-5:
        return unresolved('no translational camera baseline')
    if not np.any(mask):
        return unresolved('empty floor mask')
    geometry=[_camera_geometry(previous_calibration),_camera_geometry(current_calibration)]
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.002,edgeThreshold=15,sigma=1.)
    features=[sift.detectAndCompute(gray[0],None),sift.detectAndCompute(gray[1],mask)]
    evidence['feature_counts']=[len(k) for k,d in features]
    if any(d is None or len(d)<2 for k,d in features):
        return unresolved('insufficient floor texture')
    pairs=cv2.BFMatcher().knnMatch(features[0][1],features[1][1],k=2)
    matches=[m for m,n in pairs if m.distance<.85*n.distance]
    evidence['ratio_match_count']=len(matches)
    records=[]
    for match in matches:
        a=features[0][0][match.queryIdx].pt;b=features[1][0][match.trainIdx].pt
        triangulated=_triangulate(a,b,geometry)
        from .rgb_grounding import _triangulate as stable_triangulate
        stable=stable_triangulate({'a':a,'b':b},{'a':previous_calibration,'b':current_calibration})
        if stable is not None and triangulated is not None and triangulated[1]<.01:
            records.append({'a':a,'b':b,'distance':match.distance,'world_m':stable[0].tolist()})
    evidence['static_candidate_count']=len(records)
    evidence['stable_candidates']=records
    if len(records)<4:
        return unresolved('insufficient calibrated static correspondences')
    A=np.float32([r['a'] for r in records]);B=np.float32([r['b'] for r in records])
    homography,inliers=cv2.findHomography(A,B,cv2.RANSAC,2.)
    if homography is None:
        return unresolved('no consistent floor homography')
    evidence['homography']=homography.tolist()
    selected=[r for r,k in zip(records,inliers.ravel()) if k]
    evidence['homography_inlier_count']=len(selected)
    # Multiple SIFT scales at one image location must not vote as independent evidence.
    unique=[]
    for record in sorted(selected,key=lambda r:r['distance']):
        if all(np.linalg.norm(np.asarray(record['b'])-r['b'])>2 for r in unique):
            unique.append(record)
    if len(unique)<4:
        return unresolved('fewer than four distinct floor patches')
    records=[];rays=[]
    for record in unique:
        a,b=record['a'],record['b']
        triangulated=_triangulate(a,b,geometry)
        if triangulated is None or triangulated[1]>=.01:continue
        world,gap,ray=triangulated
        records.append({'previous_pixel':list(a),'current_pixel':list(b),'world_m':world.tolist(),'ray_gap_m':gap})
        rays.append(ray)
    evidence['matches']=records
    if len(records)<4:
        return unresolved('fewer than four refined static floor patches')
    center=np.mean([r['world_m'] for r in records],axis=0)
    B=np.asarray([r['current_pixel'] for r in records])
    # Fit slopes and offset about the measured patch center, without a stored floor height.
    def intersect(parameters,origin,direction):
        ax,by,height=parameters;normal=np.array([-ax,-by,1.])
        distance=normal@center+height
        return np.asarray(origin)+np.asarray(direction)*(distance-normal@origin)/(normal@direction)
    def residual(parameters):
        return np.asarray([_project(intersect(parameters,*ray),current_calibration)-b
                           for ray,b in zip(rays,B)]).ravel()
    fitted=least_squares(residual,[0.,0.,0.],loss='soft_l1',f_scale=.5,max_nfev=150)
    normal=np.r_[-fitted.x[:2],1.];scale=np.linalg.norm(normal);normal/=scale
    plane_offset=float(normal@center+fitted.x[2]/scale)
    reprojection=np.linalg.norm(residual(fitted.x).reshape(-1,2),axis=1)
    evidence['plane_normal_world']=normal.tolist();evidence['plane_offset_m']=plane_offset
    evidence['reprojection_errors_px']=reprojection.tolist()
    # Repeated wood patterns can match along epipolar lines at an incorrect depth.
    if normal[2]<.8 or np.max(reprojection)>1.5:
        return unresolved('matches do not support one approximately level floor plane')
    clearance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    v,u=np.unravel_index(np.argmax(clearance),clearance.shape);pixel=(float(u),float(v))
    origin,direction=_pixel_ray_world(pixel,geometry[1])
    point=intersect(fitted.x,origin,direction)
    if np.dot(point-np.asarray(origin),direction)<=0:
        return unresolved('floor intersection is behind the current camera')
    base_rotation=np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    target_base=base_rotation.T@(point-np.asarray(public_state['state.base_position']))
    evidence['chosen_pixel']=list(pixel);evidence['reason']='calibrated stereo surface plane'
    return {'status':'grounded','target_base_m':target_base.tolist(),
            'target_world_m':point.tolist(),'evidence':evidence}
