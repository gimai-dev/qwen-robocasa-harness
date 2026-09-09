"""Measured preparation and approach stages for the Qwen/tool workflow."""
from copy import deepcopy

from .joint_protocol import JOINT_LIMITS, JOINT_NAMES, JOINT_RESPONSE_SCHEMA


def readiness_required(state, receipts):
    qpos = state.get('state.arm_joint_position')
    velocity = state.get('state.arm_joint_velocity')
    if qpos is None or velocity is None:
        return False
    if qpos[3] < -1.0:
        return False
    if any(r.get('step_count', 0) > 0 and r.get('kind') in
           {'image_servo', 'cartesian_delta', 'base_action'} for r in receipts):
        return False
    ready_in_progress = any(r.get('kind')=='move_joints' and r.get('step_count',0)>0
                            and r.get('requested_targets',{}).get('joint4',0.) < -1.
                            for r in receipts)
    return ready_in_progress or max(abs(v) for v in velocity) <= .15


def ready_schema(observation_id):
    schema = deepcopy(JOINT_RESPONSE_SCHEMA)
    props = schema['properties']
    props['observation_id'] = {'const': observation_id}
    targets = props['targets']
    for name, (lower, upper) in zip(JOINT_NAMES, JOINT_LIMITS):
        targets['properties'][name] = {'type': 'number', 'minimum': lower + .02, 'maximum': upper - .02}
    targets['properties']['joint4'] = {'type': 'number', 'minimum': -2.5, 'maximum': -1.3}
    targets['properties']['gripper'] = {'const': 1.0}
    targets['required'] = ['joint2', 'joint4', 'joint6', 'gripper']
    return schema


def approach_schema(schema):
    output = deepcopy(schema)
    for branch in output.get('oneOf', [output]):
        props = branch.get('properties', {})
        if props.get('kind', {}).get('const') == 'image_servo':
            props['depth_delta_m'] = {'const': 0.0}
    return output


def prepare_controller_request(kwargs, state, receipts, history):
    stage = history[-1] if history else 'observe'
    if stage not in {'observe', 'approach'}:
        return
    if stage == 'observe' and readiness_required(state, receipts):
        kwargs['response_schema'] = ready_schema(kwargs['observation_id'])
        text = (
            'PREPARE: the measured elbow is near extension and no spatial approach has executed. '
            'Before reaching, choose a bent ready pose with coordinated shoulder joint2, elbow joint4 '
            'and wrist joint6, gripper open. Choose their values from the public robot geometry and '
            'current views; do not move the base or servo to the object yet. Keep the current milestone. '
            'If the previous ready-pose targets are still being tracked, continue those targets until '
            'the measured arm has bent. Judge readiness from measured joint angles, not your command.'
        )
    else:
        if stage == 'observe':
            kwargs['response_schema'] = approach_schema(kwargs['response_schema'])
        text = (
            'APPROACH: first align to the selected point. A stereo pixel pair already determines '
            'its 3D location; use zero additional depth during approach. Reserve insertion depth '
            'for engage after measured alignment. After the arm has reached its ready posture, '
            'select the target from current images and proceed instead of repeating ready-pose commands.'
        )
    kwargs['instruction'] += '\n\nWORKFLOW_STAGE:\n' + text


def execution_mode(packet, family, history, skill_active=False, proposal_mode=None):
    if skill_active:
        return None
    if proposal_mode is not None:
        return proposal_mode
    if not packet:
        return None
    latest=packet['latest_execution']
    failures=latest.get('observed_failures', [])
    if any(f in failures for f in ['translation_not_realized', 'servo_stalled',
                                  'approaching_joint_limit', 'moved_away_from_previous_target_estimate']):
        return {'mode':'reconfigure', 'previous_command':latest['command'], 'evidence':latest}
    if (history and history[-1]=='approach' and latest.get('outcome')=='target_estimate_reached'):
        return {'mode':'inspect_contact', 'next_milestone':'pregrasp' if family=='grasp_place' else 'engage',
                'previous_command':latest['command'], 'evidence':latest}
    return None


def repeated_recovery_command(mode, command):
    if mode is None:
        return False
    def motor(c):
        return {k:v for k,v in c.items() if k not in {'note','observation_id'}}
    return motor(mode['previous_command'])==motor(command)


def alternative_schema(observation_id, rejected_kind):
    from .joint_runner import controller_response_schema_for_observation
    schema=controller_response_schema_for_observation(observation_id)
    schema['oneOf']=[branch for branch in schema['oneOf']
                     if branch['properties']['kind'].get('const') != rejected_kind]
    return schema


def proposal_recovery_mode(records, observation_id, revision_index):
    if revision_index < 2 or not records:
        return None
    last=records[-1]
    if (last.get('observation_id')!=observation_id or not last.get('status','').startswith('rejected')
            or last.get('contradiction') not in {'stagnation','safety_bound_risk'}
            or not isinstance(last.get('draft'),dict)):
        return None
    return {'mode':'replan','previous_command':last['draft'],
            'evidence':{'proposal_rejected':last['contradiction'],'physical_execution':False}}


def apply_execution_mode(kwargs, mode):
    if mode is None:
        return
    from .joint_runner import controller_response_schema_for_observation
    import json
    # Re-open joint and Cartesian posture choices that servo-continuation schemas hide.
    kwargs['response_schema']=controller_response_schema_for_observation(kwargs['observation_id'])
    if mode['mode']=='replan':
        kwargs['response_schema']=alternative_schema(kwargs['observation_id'],mode['previous_command']['kind'])
        instruction=(
            'Repeated proposals of this action family were rejected before execution. This is planning '
            'feedback, not an observed physical failure. Choose another action family to change the '
            'approach: a joint posture, Cartesian retreat/orientation, or base repositioning as suitable. '
            'Use the measured state and current RGB to choose values; preserve the current milestone '
            'and existing safety bounds. Do not spend the remaining proposals repeating the rejected move.'
        )
    elif mode['mode']=='reconfigure':
        instruction=(
            'The executed approach failed according to measured motion. Repeating identical motor '
            'parameters is not a repair. Choose a posture change, a retreat and new approach, or '
            'a freshly identified target. If near a joint limit, first move away from that limit; '
            'do not keep pushing along the blocked ray. Keep the current milestone while recovering. '
            'Use a short note: hypothesis=<cause>;change=<actual motor change>;expect=<measured result>.'
        )
    else:
        instruction=(
            'The gripper is within 20 mm of the point you selected. This proves reaching that estimate, '
            'not contact, enclosure, or task success. Stop issuing the same approach. Inspect current RGB '
            'and enter '+mode['next_milestone']+' to orient the gripper and establish contact. '
            'If the selected point was wrong, choose a different point explicitly. Do not close merely '
            'because a geometric point was reached; check finger/object enclosure.'
        )
    kwargs['instruction'] += '\n\nMEASURED_WORKFLOW_TRANSITION:\n'+json.dumps(mode, separators=(',',':'))+'\n'+instruction
