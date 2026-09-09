"""Locate a task receiving container from current RGB and SAM masks."""
def ground_receiver(kwargs,task,folder,query):
    from .skill_driver import object_schema
    from .sam_source_grounding import ground_sam_source
    coarse=query(kwargs,f'Task: {task}. Identify the DESTINATION FIXTURE or vessel where the held source belongs. For a cabinet identify its OPEN INTERIOR SHELF, for a microwave its interior tray, for a counter its exposed top, and for a bowl its opening. Cabinets may be high in the image. Return the destination name and center of the visible receiving surface in each camera. Exclude the held source and source support.  Inspect cabinet structure: a door may be swung out to the side; a flat panel behind a small round object may be the interior back wall. Use the front shelf edge and side walls to distinguish an open cavity from a closed door; color and a round object alone do not establish a closed door. Coordinates 0–1000. Invisible views false and null.',object_schema())
    def select(instruction,images,schema):return query({**kwargs,'images':images},instruction,schema,image_roles={'left':'single_camera_original','right':'original_color_receiver_crops','wrist':'binary_receiver_masks'})
    ground=ground_sam_source(kwargs['images'],coarse,kwargs['proposal_audit_context']['camera_calibration'],kwargs['public_state'],folder,select,task=task,role='destination')
    return coarse,ground
