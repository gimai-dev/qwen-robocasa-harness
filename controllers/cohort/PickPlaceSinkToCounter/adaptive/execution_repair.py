"""Episode-local Qwen repair history grounded in executed public receipts."""
from copy import deepcopy
import math

from .action_preview import world_point

class ExecutionRepair:
    def __init__(self, family=None):
        self.family=family
        self.pending=None
        self.records=[]

    def proposed(self,command,state,landmark):
        self.pending={'command':deepcopy(command),'state':deepcopy(state),
                      'landmark':deepcopy(landmark),
                      'repair_of':self.records[-1]['observation_id'] if self.records and self.records[-1]['observed_failures'] else None}

    def observe(self,observation_id,state,receipts):
        if self.pending is None:
            return self.packet()
        pending=self.pending
        command=pending['command']
        preparing=(command.get('kind')=='move_joints' and
                   command.get('note','').startswith('milestone=observe;'))
        receipt=next((r for r in reversed(receipts) if r.get('observation_id')==command['observation_id'] and r.get('step_count',0)>0),None)
        if receipt is None:
            return self.packet()
        before=world_point(pending['state']['state.end_effector_position_relative'],pending['state'])
        after=world_point(state['state.end_effector_position_relative'],state)
        moved=math.dist(before,after)
        failures=[]
        residual=receipt.get('gripper_residual',{})
        separation=residual.get('measured_end_finger_separation')
        close=command.get('gripper')=='close' or command.get('targets',{}).get('gripper')==0
        grasping=self.family in {'grasp_place','articulated'} or command.get('note','').startswith('milestone=grasp;')
        if grasping and close and isinstance(separation,(float,int)) and separation<.0032:
            failures.append('empty_close')
        failure_status=receipt.get('failure_status')
        if failure_status not in (None,'none','ok','success'):
            failures.append(str(failure_status))
        target=receipt.get('stereo_target_base_m')
        target_source='executed_stereo_receipt' if target is not None else None
        if target is None:
            target=(pending['landmark'] or {}).get('estimated_landmark_position_robot_base_m')
            target_source='qwen_landmark' if target is not None else None
        reduction=None
        distance_after=None
        if target is not None:
            fixed_target=world_point(target,pending['state'])
            distance_after=math.dist(after,fixed_target)
            reduction=math.dist(before,fixed_target)-distance_after
            if reduction<-.005 and not preparing:
                failures.append('moved_away_from_previous_target_estimate')
        requested_translation=receipt.get('resolved_translation_m', command.get('translation_m'))
        expected_motion=math.sqrt(sum(v*v for v in requested_translation)) if requested_translation is not None else None
        if expected_motion is not None and expected_motion>.005 and moved<.002:
            failures.append('translation_not_realized')
        if (command.get('kind')=='image_servo' and distance_after is not None and distance_after>.02
                and moved<.002 and self.records and self.records[-1]['command'].get('kind')=='image_servo'
                and self.records[-1]['eef_motion_norm_m']<.002):
            failures.append('servo_stalled')
        from .joint_protocol import JOINT_LIMITS
        q_before=pending['state'].get('state.arm_joint_position')
        q_after=state.get('state.arm_joint_position')
        if q_before is not None and q_after is not None:
            for a,b,(lower,upper) in zip(q_before,q_after,JOINT_LIMITS):
                if (upper-b<.06 and b-a>.005) or (b-lower<.06 and a-b>.005):
                    failures.append('approaching_joint_limit')
                    break
        outcome='observed_failure' if failures else (
            'target_estimate_reached' if not preparing and distance_after is not None and distance_after<=.02 else (
                'measured_approach_progress' if not preparing and reduction is not None and reduction>.005 else 'effect_requires_visual_interpretation'))
        self.records.append({'observation_id':command['observation_id'],'after_observation_id':observation_id,
            'command':command,'qwen_plan':command.get('note',''),'repair_of':pending['repair_of'],
            'eef_motion_world_m':[b-a for a,b in zip(before,after)],'eef_motion_norm_m':moved,
            'distance_reduction_to_previous_estimate_m':reduction,'finger_separation_m':separation,
            'distance_to_previous_estimate_m':distance_after,'target_evidence_source':target_source,
            'expected_translation_norm_m':expected_motion,
            'observed_failures':failures,'outcome':outcome,'task_success_proven':False,
            'evidence_status':'executed_public_feedback'})
        self.pending=None
        return self.packet()

    def packet(self):
        if not self.records:
            return None
        return {'latest_execution':deepcopy(self.records[-1]),'recent_repairs':deepcopy(self.records[-3:]),
                'instruction':'Use the actual executed effect to revise your next action. In your existing note, after the required milestone prefix, write hypothesis=what physically failed; change=what you now do differently; expect=the visible or measured effect to check next. An empty close calls for checking object identity and enclosure in current RGB, not merely repeating close. Positive distance reduction confirms approach to the previous estimate only. Decide all movement values yourself using current observations. Treat a strategy as useful only after its predicted effect is observed; task success still comes from the environment.'}
