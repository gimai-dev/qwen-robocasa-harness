# H1 gallery coverage

H1 promises a crop for every supplied region, but `_gallery` sliced the list to12 while the perception layer can return14. The screen actually reached13–14 regions on44 decisions across three starts: CounterToSink seed0 (5 decisions), StoveToCounter seed1 (26), and StoveToCounter seed2 (13). The Sink start was an official task success. Its result must be retained alongside the two failed starts when replacing the affected attempts.

The gallery now includes the entire delivered region list. Its existing layout expands to a fourth row when necessary; the number of model images is unchanged. Inputs for recorded observations with at most12 regions are unchanged. The shared perception, controller, prompts, and decoding settings are unchanged.

A regression puts a distinctive red object only in the14th region. The old renderer omitted all400 red pixels; the repaired renderer retains them. All49 repair regressions pass locally and on h200-4. Actual14-region initial observations from Stove seeds1 and2 render at512×624; the final row is visible and its labels remain legible. No robot-motion probe is needed for this image-only repair.

After the original36-run screen and its inspections finish, retain the three exposed H1 attempts, their launch logs, and cached inspections. Rerun those same pinned starts at the repaired revision. Reuse the other six H1 runs and every other condition, which did not execute the changed image path. Keep the clean-derived H6/H7 banks frozen.

Separate perception finding: the initial Drawer seed0 whisk is visible in RGB, but none of its21 retained right-view SAM masks isolates the whisk. The six-entry unpaired cap did not remove a dedicated whisk mask in this observation. Region r4 also pairs a large right-view counter mask with a small wrist fragment; the gallery selects that fragment. Raw rejected SAM proposals were not retained, so generation versus filtering is not isolated. Gallery coverage repair does not resolve these perception limitations, which require a separate experiment.
