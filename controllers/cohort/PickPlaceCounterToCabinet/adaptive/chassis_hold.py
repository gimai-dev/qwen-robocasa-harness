"""Episode-persistent chassis hold using only public robot XY and yaw."""
from dataclasses import dataclass
import math

XY_GAIN = 10.0  # 20 Hz: at most half the position error per selected-axis tick.
YAW_GAIN = 4.0
XY_DEADBAND_M = .0025
YAW_DEADBAND_RAD = math.radians(.5)
NORMALIZED_LIMIT = .25
VELOCITY_SCALES = (1., 1., 1.5)  # Installed Omron velocity-actuator ctrlranges.


def public_chassis_pose(state):
    position = state['state.base_position']
    x, y, z, w = map(float, state['state.base_rotation'])
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return (float(position[0]), float(position[1]), yaw)


def _rotate(angle, vector):
    c, s = math.cos(angle), math.sin(angle)
    x, y = vector
    return (c*x-s*y, s*x+c*y)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class ChassisHold:
    target_world_xy_m: tuple[float, float]
    target_yaw_rad: float
    reset_yaw_rad: float

    @classmethod
    def from_public_state(cls, state):
        x, y, yaw = public_chassis_pose(state)
        return cls((x,y), yaw, yaw)

    def reset_after_base_action(self, state):
        x, y, yaw = public_chassis_pose(state)
        self.target_world_xy_m = (x,y)
        self.target_yaw_rad = yaw
        # reset_yaw is the inner controller's episode frame, not this target.

    def errors(self, state):
        x, y, yaw = public_chassis_pose(state)
        return ([self.target_world_xy_m[0]-x, self.target_world_xy_m[1]-y],
                _wrap(self.target_yaw_rad-yaw))

    def correction(self, state):
        error, yaw_error = self.errors(state)
        current_yaw = public_chassis_pose(state)[2]
        if math.hypot(*error) <= XY_DEADBAND_M:
            error = [0.,0.]
        initial_xy = _rotate(-self.reset_yaw_rad, error)
        # Installed MobileBaseJointVelocityController applies R(-delta_yaw).
        # Invert that map before choosing its one permitted input axis.
        input_xy = _rotate(current_yaw-self.reset_yaw_rad, initial_xy)
        candidate = [XY_GAIN*input_xy[0], XY_GAIN*input_xy[1],
                     0. if abs(yaw_error)<=YAW_DEADBAND_RAD else YAW_GAIN*yaw_error/VELOCITY_SCALES[2]]
        index = max(range(3), key=lambda i:abs(candidate[i]))
        motion = [0.,0.,0.]
        motion[index] = max(-NORMALIZED_LIMIT, min(NORMALIZED_LIMIT,candidate[index]))
        return motion

    def receipt(self, state, corrections, *, reset=False):
        error, yaw_error = self.errors(state)
        return {'event':'reset_after_base_action' if reset else 'hold_during_arm',
                'target_world_xy_m':list(self.target_world_xy_m),
                'target_yaw_rad':self.target_yaw_rad, 'reset_yaw_rad':self.reset_yaw_rad,
                'xy_gain_s_inv':XY_GAIN, 'yaw_gain_s_inv':YAW_GAIN,
                'xy_deadband_m':XY_DEADBAND_M, 'yaw_deadband_rad':YAW_DEADBAND_RAD,
                'normalized_limit':NORMALIZED_LIMIT,
                'applied_corrections':[list(v) for v in corrections],
                'final_error_world_xy_m':error, 'final_error_yaw_rad':yaw_error}


def validate_chassis_receipt(value, step_count):
    """Keep the added public receipt bounded to the actual actuation contract."""
    fields={'event','target_world_xy_m','target_yaw_rad','reset_yaw_rad',
            'xy_gain_s_inv','yaw_gain_s_inv','xy_deadband_m','yaw_deadband_rad',
            'normalized_limit','applied_corrections','final_error_world_xy_m','final_error_yaw_rad'}
    if not isinstance(value,dict) or set(value)!=fields:
        raise ValueError('chassis hold receipt fields drifted')
    if value['event'] not in {'hold_during_arm','reset_after_base_action'}:
        raise ValueError('chassis hold event is invalid')
    for key,width in [('target_world_xy_m',2),('final_error_world_xy_m',2)]:
        if len(value[key])!=width or not all(math.isfinite(float(v)) for v in value[key]):
            raise ValueError('chassis hold pose is invalid')
    for key in fields-{'event','target_world_xy_m','final_error_world_xy_m','applied_corrections'}:
        if not math.isfinite(float(value[key])):
            raise ValueError('chassis hold scalar is invalid')
    rows=value['applied_corrections']
    expected=step_count if value['event']=='hold_during_arm' else 0
    if len(rows)!=expected:
        raise ValueError('chassis correction count differs from actuation count')
    for row in rows:
        if len(row)!=3 or any(not math.isfinite(float(v)) or abs(v)>.25+1e-12 for v in row) or sum(abs(v)>1e-12 for v in row)>1:
            raise ValueError('chassis correction exceeds bounded one-axis actuation')
    return value
