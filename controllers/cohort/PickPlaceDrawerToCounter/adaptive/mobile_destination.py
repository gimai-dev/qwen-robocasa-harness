"""Locate a task receiving container from current RGB and SAM masks."""
def ground_receiver(kwargs,task,folder,query):
    from .skill_driver import object_schema
    from .sam_source_grounding import ground_sam_source
    coarse=query(kwargs,f'Task: {task}. Identify the DESTINATION FIXTURE or vessel where the held source belongs. For a cabinet identify its OPEN INTERIOR SHELF, for a microwave its interior tray, for a counter its exposed top, and for a bowl its opening. Cabinets may be high in the image. Return the destination name and center of the visible receiving surface in each camera. Exclude the held source and source support. Coordinates 0–1000. Invisible views false and null.',object_schema())
    def select(instruction,images,schema):return query({**kwargs,'images':images},instruction,schema,image_roles={'left':'single_camera_original','right':'original_color_receiver_crops','wrist':'binary_receiver_masks'})
    ground=ground_sam_source(kwargs['images'],coarse,kwargs['proposal_audit_context']['camera_calibration'],kwargs['public_state'],folder,select,task=task,role='destination')
    if coarse.get('target','').lower()=='counter':
        import cv2
        from pathlib import Path
        from .stereo_surface import stereo_surface_point
        surfaces=[]
        for view,trace in ground.get('evidence',{}).get('selections',{}).items():
            response=trace.get('response',{})
            selected=next((c for c in ground['evidence']['sam']['views'][view]['candidates'] if c['id']==response.get('mask_id')),None)
            if not response.get('visible') or selected is None:continue
            mask=cv2.imread(selected['mask_path'],0)
            for other in ('left','right','wrist'):
                if other==view:continue
                result=stereo_surface_point(kwargs['images'][other],kwargs['images'][view],kwargs['proposal_audit_context']['camera_calibration'][other],kwargs['proposal_audit_context']['camera_calibration'][view],mask,kwargs['public_state'])
                if result['status']=='grounded':surfaces.append(result)
        if surfaces:
            ground=max(surfaces,key=lambda x:len(x['evidence']['matches']))
        else:
            ground={'status':'unresolved','target_base_m':None,'evidence':{**ground.get('evidence',{}),'reason':'visible counter needs corresponding surface points rather than mask centroids'}}
    return coarse,ground
