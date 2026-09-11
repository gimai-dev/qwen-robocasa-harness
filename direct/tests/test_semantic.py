import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.semantic import TOKENS, parse_token, Interpreter

STATE = {"tcp_world_position_m": [1.0, 2.0, 1.2], "tcp_world_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
         "base_world_yaw_rad": 0.0, "gripper_width_m": 0.078}


def test_vocabulary():
    assert TOKENS == ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
                      "ROTATE_CW", "ROTATE_CCW", "GRASP", "RELEASE", "DONE",
                      "BASE_FWD", "BASE_BACK", "BASE_LEFT", "BASE_RIGHT")


def test_parse_token_strips_and_validates():
    assert parse_token(" MV_UP\n") == "MV_UP"
    try:
        parse_token("MOVE UP"); assert False
    except ValueError:
        pass


def test_move_fine_is_two_cm_in_base_frame():
    a = Interpreter().to_action("MV_FWD", STATE, fine=True)
    assert a.kind == "ee" and abs(a.position_m[0] - 1.02) < 1e-9 and abs(a.position_m[1] - 2.0) < 1e-9


def test_move_uses_base_heading():
    s = dict(STATE, base_world_yaw_rad=math.pi / 2)
    a = Interpreter().to_action("MV_FWD", s, fine=False)
    assert abs(a.position_m[0] - 1.0) < 1e-9 and abs(a.position_m[1] - 2.04) < 1e-9


def test_rotate_changes_only_yaw_about_tool_axis():
    a = Interpreter().to_action("ROTATE_CW", STATE, fine=True)
    assert a.kind == "ee" and a.position_m == (1.0, 2.0, 1.2) and tuple(a.quat_xyzw) != (1.0, 0.0, 0.0, 0.0)


def test_gripper_tokens_and_done():
    i = Interpreter()
    assert i.to_action("GRASP", STATE, fine=True).kind == "hold" and i.to_action("GRASP", STATE, fine=True).gripper == 0
    assert i.to_action("RELEASE", STATE, fine=True).gripper == 1
    assert i.to_action("DONE", STATE, fine=True) is None


def test_base_tokens():
    a = Interpreter().to_action("BASE_LEFT", STATE, fine=True)
    assert a.kind == "base" and a.axis == "y" and a.velocity == 0.5
