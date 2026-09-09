"""Reuse two recent static-source observations when one frame lacks stereo."""
import cv2
import numpy as np
from .sam_source_grounding import ground_source_masks

def update_source_history(history,ground,coarse,calibration,state):
    evidence=ground.get('evidence',{});views={}
    for view,trace in evidence.get('selections',{}).items():
        response=trace.get('response',{})
        proposals=evidence.get('sam',{}).get('views',{}).get(view,{}).get('candidates',[])
        selected=next((c for c in proposals if c['id']==response.get('mask_id')),None)
        if response.get('visible') and selected is not None:
            views[view]={'mask':cv2.imread(selected['mask_path'],0)>0,'calibration':calibration[view]}
    candidates=[];locations=[]
    if ground['status']!='grounded':
        for age,old in enumerate(history):
            for a,va in old.items():
                for b,vb in views.items():
                    baseline=np.linalg.norm(np.asarray(va['calibration']['camera_position_world_m'])-vb['calibration']['camera_position_world_m'])
                    if baseline<.03:continue
                    result=ground_source_masks({'left':va['mask'],'right':vb['mask']},coarse,{'left':va['calibration'],'right':vb['calibration']},state)
                    for location in result.get('evidence',{}).get('localization_candidates',[]):
                        locations.append({**location,'temporal_views':[a,b]})
                    if result['status']=='grounded':
                        result['evidence']['temporal_views']={'previous_index':age,'previous_camera':a,'current_camera':b,'camera_baseline_m':float(baseline)}
                        candidates.append(result)
    history.append(views);history[:]=history[-2:]
    if candidates:
        selected=min(candidates,key=lambda c:c['ray_gap_m']);selected['evidence']['current_frame_status']=ground['status'];return selected
    ground.setdefault('evidence',{}).setdefault('localization_candidates',[]).extend(locations)
    return ground
