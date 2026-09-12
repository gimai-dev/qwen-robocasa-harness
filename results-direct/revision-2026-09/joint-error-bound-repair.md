# Executable bounds in joint error feedback

The Stage3 joint-short CounterToDrawer seed0 run at `ef3e956` rejected joint6=0 at decision26 and reported a minimum of0.002. At decision27, Qwen used0.002 and was rejected again. The actual lower bound is0.0025. Although the main guide had already been corrected, the decoder's error feedback still rounded some limits outward.

The error message now rounds its displayed lower bound upward and upper bound downward at three decimals, consistent with executable guide endpoints. The validation condition and physical joint limits are unchanged. A regression parses every joint's error message and sends both advertised endpoints back through the real decoder: four endpoints failed before repair and all14 pass afterward. The separate two-guide endpoint test still passes all28 cases. All48 repair regressions pass locally and on h200-4.

Retain completed attempts that received these misleading joint-bound errors separately and rerun only those starts. Runs that never received a joint-bound error, including unaffected EE/full runs, remain eligible. This follow-up changes error feedback only; no simulator probe is needed because executed targets, rates, and limits are unchanged.
