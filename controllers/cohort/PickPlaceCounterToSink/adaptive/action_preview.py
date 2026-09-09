"""Kinematic previews from the controller's own commands and public geometry."""
from collections.abc import Mapping
from io import BytesIO
import math

from PIL import Image, ImageDraw, UnidentifiedImageError

from .image_servo import _matvec, _rotation_xyzw
from .joint_runner import prepare_joint_mailbox
from .panda_embodiment import panda_fk, project_world_point

CURRENT = '#369ca5'
PROPOSED = '#476ee8'
TARGET = '#bd793e'

def world_point(point, state):
    rotated = _matvec(_rotation_xyzw(state['state.base_rotation']), point)
    return [a+b for a,b in zip(rotated,state['state.base_position'])]

def preview_geometry(draft, state, calibration, current_gripper):
    current = list(state['state.end_effector_position_relative'])
    rotation = _rotation_xyzw(state['state.end_effector_rotation_relative'])
    predicted = current[:]
    predicted_rotation = rotation
    gripper = current_gripper
    if draft and draft.get('kind') in {'move_joints','cartesian_delta','image_servo'}:
        mailbox,_ = prepare_joint_mailbox(
            draft,source='controller',observation_id=draft['observation_id'],
            current_qpos=state['state.arm_joint_position'],current_gripper=current_gripper,
            public_state=state,camera_calibration=calibration,remaining_actions=450,sequence=0)
        before=panda_fk(state['state.arm_joint_position'])
        after=panda_fk(mailbox['endpoint'])
        predicted=[current[i]+after.position_m[i]-before.position_m[i] for i in range(3)]
        change=[[sum(after.rotation_matrix[i][k]*before.rotation_matrix[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        predicted_rotation=[[sum(change[i][k]*rotation[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        gripper=mailbox['gripper_open']
    return {
        'current_eef_robot_base_m':current,
        'predicted_eef_robot_base_m':predicted,
        'predicted_eef_world_m':world_point(predicted,state),
        'current_rotation_robot_base':rotation,
        'predicted_rotation_robot_base':predicted_rotation,
        'predicted_gripper_open':gripper,
        'delta_robot_base_m':[b-a for a,b in zip(current,predicted)],
        'prediction_scope':'kinematic_endpoint_not_contact_or_collision',
        'base_motion_previewed':False,
    }

def _glyph(position,rotation,opening):
    half_width=.006+.034*opening
    def at(local):
        delta=_matvec(rotation,local)
        return [a+b for a,b in zip(position,delta)]
    return [(at((0,-half_width,-.045)),at((0,-half_width,0))),
            (at((0,half_width,-.045)),at((0,half_width,0))),
            (at((0,-half_width,-.045)),at((0,half_width,-.045))),
            (at((0,0,-.045)),at((0,0,-.065)))]

def render_preview(display_images,raw_images,state,calibration,draft,current_gripper,
                   landmark=None,previous_wrist=None):
    """Append geometry panels; original display pixels and image slots stay intact."""
    try:
        original=Image.open(BytesIO(display_images['wrist'])).convert('RGB')
        raw=Image.open(BytesIO(raw_images['wrist'])).convert('RGB')
    except (UnidentifiedImageError,KeyError):
        return display_images,None
    try:
        geometry=preview_geometry(draft,state,calibration,current_gripper)
        error=None
    except (ValueError,TypeError,KeyError) as exc:
        geometry=preview_geometry(None,state,calibration,current_gripper)
        error=str(exc)
    width=max(original.width,768)
    canvas=Image.new('RGB',(width,original.height+256),'#f4f3ef')
    canvas.paste(original,(0,0))
    draw=ImageDraw.Draw(canvas)
    if previous_wrist and original.width<=512:
        previous=Image.open(BytesIO(previous_wrist)).convert('RGB')
        previous.thumbnail((220,220))
        canvas.paste(previous,(width-240,24))
        draw.text((width-240,8),'BEFORE last executed action',fill='#343434')
    y=original.height
    raw.thumbnail((256,224))
    canvas.paste(raw,(0,y+28))
    draw.text((8,y+6),'CURRENT wrist + pose projection',fill='#343434')
    current=geometry['current_eef_robot_base_m']
    predicted=geometry['predicted_eef_robot_base_m']
    target=landmark.get('estimated_landmark_position_robot_base_m') if isinstance(landmark,Mapping) else None
    if original.width<=256:
        lines=['ROBOT BASE FRAME (meters)',
               'Current: '+', '.join(f'{v:+.3f}' for v in current),
               'Preview: '+', '.join(f'{v:+.3f}' for v in predicted),
               'Delta:   '+', '.join(f'{v:+.3f}' for v in geometry['delta_robot_base_m'])]
        if target is not None:
            lines.extend(['Target:  '+', '.join(f'{v:+.3f}' for v in target),
                          f'Distance now: {math.dist(current,target):.3f}m',
                          f'After preview: {math.dist(predicted,target):.3f}m'])
        for row,text in enumerate(lines):
            draw.text((264,16+row*22),text,fill='#343434')
    glyphs=[(_glyph(current,geometry['current_rotation_robot_base'],current_gripper),CURRENT)]
    if draft and error is None:
        glyphs.append((_glyph(predicted,geometry['predicted_rotation_robot_base'],geometry['predicted_gripper_open']),PROPOSED))
    camera=calibration.get('wrist')
    if isinstance(camera,Mapping):
        for segments,color in glyphs:
            for a,b in segments:
                try:
                    points=[project_world_point(world_point(p,state),camera) for p in (a,b)]
                except ValueError:
                    continue
                if all(p['visible'] for p in points):
                    draw.line([(p['u_px']*raw.width/camera['image_width_px'],y+28+p['v_px']*raw.height/camera['image_height_px']) for p in points],fill=color,width=2)
    points=[current,predicted]+([target] if target is not None else [])
    span=max(.14,max(max(p[i] for p in points)-min(p[i] for p in points) for i in range(3))+.10)
    center=[(max(p[i] for p in points)+min(p[i] for p in points))/2 for i in range(3)]
    for index,(a,b,label) in enumerate([(0,1,'BASE XY'),(0,2,'BASE XZ')]):
        x=256+index*256
        draw.rectangle((x,y,x+255,y+255),outline='#cccccc')
        draw.text((x+8,y+8),label+'  meters',fill='#343434')
        draw.text((x+224,y+130),'+X',fill='#343434')
        draw.text((x+116,y+28),'+'+('Y' if b==1 else 'Z'),fill='#343434')
        def plot(point):
            return (x+128+(point[a]-center[a])*200/span,y+140-(point[b]-center[b])*200/span)
        for offset in [-.5,0,.5]:
            line=x+128+offset*200
            draw.line((line,y+40,line,y+240),fill='#dddddd')
        for segments,color in glyphs:
            for p,q in segments:
                draw.line((plot(p),plot(q)),fill=color,width=3)
        draw.line((plot(current),plot(predicted)),fill=PROPOSED,width=2)
        if target is not None:
            tx,ty=plot(target); draw.ellipse((tx-5,ty-5,tx+5,ty+5),outline=TARGET,width=2)
        draw.text((x+8,y+240),f'width={span:.3f}m',fill='#343434')
    if original.width<=512 and not previous_wrist:
        for line,text in enumerate(['CYAN: measured gripper','BLUE: Qwen proposal','ORANGE: Qwen target estimate','Glyph shows pose, not collision','No physics step in preview']):
            draw.text((width-245,15+line*22),text,fill='#343434')
    out=BytesIO();canvas.save(out,format='PNG')
    packet={'geometry':geometry,'preview_error':error,'landmark':landmark,
            'layout':{'original_wrist_rectangle':[0,0,original.width,original.height],
                      'preview_row_y':y,'panel_width':256,'image_size':list(canvas.size)},
            'meaning':'Top-left preserves the prior wrist display. Bottom: current wrist with pose glyph, robot-base XY and XZ. Cyan is measured, blue is your proposed kinematic endpoint, orange is a Qwen-selected geometric estimate. Axes use meters. Preview is not executed; inspect RGB for object identity and obstacles. Reissuing a rejected action should follow a changed physical hypothesis. Base pulses have no predicted endpoint here.'}
    return {**display_images,'wrist':out.getvalue()},packet
