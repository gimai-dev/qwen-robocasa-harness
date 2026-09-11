"""Direct numerical Qwen control of the RoboCasa PandaOmron with a shared executor.

Frozen Qwen emits numerical end-effector or joint targets; the executor performs
only IK, interpolation, tracking and bound handling. Experimental conditions
(clean and H1-H8) change what Qwen sees or how its numbers are interpreted, not
who chooses the action.
"""
