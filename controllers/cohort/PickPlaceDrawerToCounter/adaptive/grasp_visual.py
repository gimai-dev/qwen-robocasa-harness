"""Three actual steps around closure, cropped using public gripper pixels."""
import io,json
from pathlib import Path
from PIL import Image

def temporal_grasp_images(samples,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    images={}
    for slot,label,sample in zip(('left','right','wrist'),('before_close','after_close','after_lift'),samples):
        path=folder/(label+'.png');path.write_bytes(sample['images']['wrist']);images[slot]=path.read_bytes()
    metadata={'camera':'wrist','steps':['before_close','after_close','after_lift'],'source':'three actual wrist images; compare source relative to fingers as background changes'}
    (folder/'context.json').write_text(json.dumps(metadata,indent=2))
    return images,metadata

def grasp_check_instruction(task,target,camera,lift,gap):
    return (f'Task: {task} Verify the pickup of the task source {target}. '
        'These are three successive WRIST camera images: BEFORE closing, AFTER closing, AFTER lifting. '
        'The camera moves with the gripper. Compare the source BETWEEN THE FINGERS at the bottom of each image against the changing support surface. '
        'Does the same task source remain enclosed by the fingers after lifting? '
        f'Finger separation remains {gap:.3f}m; commanded lift was {lift:.3f}m. '
        'Describe actual visual change; do not infer from the gripper command alone.')
