# Joint-guide follow-up

The first Stage3 joint-short drawer rollout emitted joint6=0 at decision7, following a prompt that advertised `[0.00,3.73]`. The decoder rejected the command because its actual margin-adjusted lower bound is0.0025. The advertised endpoints for joints1,3,5,7 also rounded outside their accepted bounds. This is a prompt/action-contract defect.

The short and full joint guides now use inward-rounded three-decimal ranges. The regression passes each advertised endpoint to the real `decode_action` function. It failed18 endpoint subcases before the repair; all28 endpoint subcases pass after it. The complete44-test repair suite passes locally and on h200-4.

The scalar downward-orientation guide also now states its zero-roll condition: when joint3=joint5=0, joint2−joint4−joint6=0 makes the TCP approach axis point down. The existing, physically checked forward-kinematics model gives tool+z=[0,0,-1] for `[0,0,0,-1,0,1,0]` and for `[0.4,0.2,0,-1.2,0,1.4,0.6]`. Changing the latter to joint3=0.5 and joint5=0.3, while preserving the scalar relation, tilts that axis about22degrees from down. The previous unqualified rule did not apply to general joint configurations.

Only the two joint prompts and their contract regression change. The shared executor, observation pipeline, EE prompts, and completed Stage2 comparisons are unchanged. Early joint attempts at88243a1 are retained separately and the joint conditions are restarted. The two completed EE-full attempts from that revision remain eligible for Stage3 because their implementation and prompts are identical.

A fresh read-only review found no actionable defect in the ranges, zero-roll equation, or endpoint regression. It did not duplicate the parent tests or run additional model calls.
