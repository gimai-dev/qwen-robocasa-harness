"""Detect an observed held object disappearing from a closed gripper."""

CARRY_STAGES={'mixer_outward','mixer_orient','mixer_raise','mixer_side',
              'mixer_front','receiver_raise','receiver_base','transfer','lower_place','inspect_place','locate_place'}

def lost_during_carry(stage,confirmed_gap_m,gripper_qpos):
    if stage not in CARRY_STAGES or confirmed_gap_m is None:
        return False
    gap=abs(gripper_qpos[0]-gripper_qpos[1])
    # Actual empty Panda closure is about1mm; a held food object is much wider.
    # Require both near-empty closure and a large collapse from its lifted grip.
    return gap<.003 and gap<.25*confirmed_gap_m
