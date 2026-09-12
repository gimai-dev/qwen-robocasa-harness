# Match the H7 prompt to the verified bank

The new clean bank contains two overlapping local grasp/lift fragments from one StoveToCounter seed2 episode. Its action poses and measured grasp TCP are absolute source-scene coordinates. The inherited H7 prompt described these as relative TCP/object geometry, although no such offsets are stored. The prompt now states the actual representation and distinguishes verified local lift from full-task success. Qwen still generates new targets from the current observation. No object truth was added to the bank or policy.

This is a text-only correction for H7 before its independent applicability check. No previous campaign run used H7. The stored bank fields were compared directly with the corrected description; no new physics or broad test run is needed for this wording change. Other harnesses, prompts, request accounting, and the shared executor are unchanged.
