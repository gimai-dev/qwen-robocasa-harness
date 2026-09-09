"""Success demonstrations and post-episode lessons consumed by later episodes."""
import json
import math


def extract_card(result,evidence):
    if (result.get('terminal_outcome') or {}).get('success') is not True:
        raise ValueError('skill donor requires official success')
    events=evidence['events']
    # An object can slip into the basin and satisfy the official task while
    # the policy stops before release. Keep that outcome as a lesson; it is
    # not a demonstrated controlled release/withdrawal parameter donor.
    released=any(e['kind']=='observation' and e.get('stage')=='release' for e in events)
    withdrew=any(e['kind']=='pose_plan' and e.get('stage')=='leave' and e['plan'].get('waypoints') for e in events)
    if not (released and withdrew):
        return None
    hypothesis=next(e for e in events if e['kind']=='grasp_hypothesis')
    plans={e['stage']:e['plan']['waypoints'][-1] for e in events
           if e['kind']=='pose_plan' and e['plan'].get('waypoints')}
    grasp=plans['descend']['target_position_m']
    pre=plans['pregrasp']['target_position_m']
    close=next(e for e in events if e['kind']=='observation' and e['stage']=='close')
    release=next(e for e in reversed(events) if e['kind']=='observation' and e['stage']=='release')
    placement=next(e for e in events if e['kind']=='placement_geometry')
    side=next((e for e in events if e['kind']=='side_grasp_orientation'),None)
    height=next((e for e in events if e['kind']=='grasp_height'),None)
    front_release=any(e['kind']=='intentional_release' for e in events)
    parameters={
        'approach':'side_horizontal' if side else 'direct',
        'depth_offset_m':hypothesis['depth_offset_m'],
        'pregrasp_distance':height['nominal_pregrasp_distance_m'] if height else math.dist(pre,grasp),
        'lift_distance':plans['lift']['target_position_m'][2]-close['position'][2],
        'grasp_down_angle':side['down_angle_deg'] if side else 90.,
        'release_clearance':placement['release_point_base_m'][2]-placement['floor_center_base_m'][2],
        'withdraw_distance':plans['leave']['target_position_m'][2]-release['position'][2],
    }
    if side:
        drop=plans['side_drop']['target_position_m']
        parameters.update(side_offset_m=abs(drop[1]-grasp[1]),
                          side_clearance=drop[2]-grasp[2],
                          staging_rise=plans['side_clear']['target_position_m'][2]-grasp[2],
                          approach_height=plans['side_clear']['target_position_m'][2])
        if placement.get('route')=='vertical rise then transfer above bowl rim':
            parameters['transfer_clearance']=placement['transit_point_base_m'][2]-placement['floor_center_base_m'][2]
    if front_release:
        selected=next(e for e in events if e['kind']=='skill_selected')
        # The route consumes a minimum clearance, not its realized release z.
        parameters['release_clearance']=selected['parameters']['release_clearance']
        # Earlier native records used the fixed route defaults, or the
        # selected route card; new events store the commanded values directly.
        parameters['sink_front_offset_m']=placement.get('sink_front_offset_m',selected['parameters'].get('sink_front_offset_m',.12))
        parameters['sink_withdraw_distance_m']=placement.get('sink_withdraw_distance_m',selected['parameters'].get('sink_withdraw_distance_m',.10))
        parameters['withdraw_distance']=parameters['sink_withdraw_distance_m']
    parameters={k:round(v,6) if isinstance(v,(int,float)) else v for k,v in parameters.items()}
    return {'id':result['task']+'-seed'+str(result['seed']),
            'source_task':result['task'],'source_seed':result['seed'],
            **({'destination_route':'sink_front_release','observed_pregrasp_distance_m':math.dist(pre,grasp),'observed_release_height_m':placement['release_point_base_m'][2]-placement['floor_center_base_m'][2]} if front_release else {}),
            'official_success':True,'parameters':parameters,
            'source_events':['grasp_hypothesis','pregrasp','descend','close','lift','placement_geometry','leave'],
            'scope':'Parameters observed in one successful scene; re-localize and replan in a new scene.'}


def record_episode(bank,result,evidence,reflection,run_id):
    success=(result.get('terminal_outcome') or {}).get('success')
    if type(success) is not bool:
        raise ValueError('memory writeback requires official episode outcome')
    lesson={'source_run':run_id,'task':result['task'],'seed':result['seed'],
            'official_success':success,'observed_stop':result['status'],
            'reflection':reflection}
    bank['lessons'].append(lesson)
    if success:
        if any(e.get('kind')=='grasp_hypothesis' for e in evidence.get('events',[])):
            card=extract_card(result,evidence)
            if card is None:
                lesson['skill_status']='not_promoted: official task success without demonstrated release and withdrawal'
                return bank
        else:
            card={'id':run_id,'source_task':result['task'],'source_seed':result['seed'],
                  'official_success':True,'steps':reflection.get('skill_steps',[]),
                  'scope':'Successful episode summarized by Qwen; steps guide future proposals.'}
        card['source_run']=run_id
        card['lesson']=reflection.get('lesson')
        card['applicability']=reflection.get('applicability')
        # Repeated episodes update this task/seed skill; their individual
        # outcomes remain in lessons rather than duplicating the same ID.
        bank['skills']=[old for old in bank['skills'] if old['id']!=card['id']]
        bank['skills'].append(card)
    return bank


def retrieval_context(bank):
    return json.dumps({'successful_skills':bank['skills'],
                       'parameter_units':{'grasp_down_angle':'degrees','motion_distances':'metres'},
                       'episode_lessons':bank['lessons'],
                       'use':'Use relevant experience to choose a new proposal. Reground image targets. '
                             'Failure reflections are hypotheses, not successful trajectories. '
                             'Do not repeat an approach that failed without a concrete changed dimension.'},
                      ensure_ascii=False,separators=(',',':'))
