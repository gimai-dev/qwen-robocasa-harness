# Reproduction and environment

## Scope of the release

This is a source-and-evidence release of the completed ten-task experiment. Controller snapshots are preserved separately so that recipe-specific behavior is inspectable. The path-preparation utility maps those snapshots and memory files into a new checkout. It does not install or modify the simulator, model service, operating-system configuration, or controller algorithms.

The existing experiment host is `h200-4`. Native launches use:

| Component | Existing path or setting |
|---|---|
| Python | `/home/jli/work/robocasa-inspect-official/.venv/bin/python`, Python 3.11.16 |
| Simulator integration root | `/home/jli/work/robocasa-inspect-official` |
| RoboCasa source | `robocasa/` under the integration root |
| robosuite source | `robosuite/` under the integration root |
| Shared integration module | `robocasa_inspect/` under the integration root; source included in `runtime/` |
| Asset/cache state | `/home/jli/state/robocasa-inspect-official` |
| Asset manifest | `/home/jli/state/robocasa-inspect-official/assets/content-manifest.json` |
| NVIDIA EGL files | `/home/jli/state/robocasa-inspect-official/nvidia-egl-580.173.02/rootfs` |
| SAM Python | `/home/jli/state/qwen-rgb-sam2/.venv/bin/python` |
| SAM checkpoint | `/home/jli/state/qwen-rgb-sam2/sam2.1_hiera_small.pt` |
| Qwen endpoint | `http://127.0.0.1:8002/v1` |
| Qwen identity | `/home/jli/state/panda-qwen38/identity.json` |
| Qwen server metadata | `/home/jli/state/panda-qwen38/server-attestation.json` |
| Qwen API token | `/home/jli/state/panda-qwen38/api-token` |

The simulator child is launched through the existing `sudo -n`, `unshare`, and `setpriv` setup with UID/GID 1001 and the configured GPU-access groups. These are original launcher requirements, not new publication requirements. The model token and private service configuration are not included in this repository.

## Recorded versions and local simulator changes

| Dependency | Observed version / revision |
|---|---|
| RoboCasa | 1.0.1; `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` |
| robosuite | 1.5.2; `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| MuJoCo | 3.3.1 |
| NumPy | 2.2.5 |
| SciPy | 1.15.3 |
| PyTorch in simulator environment | 2.7.1 |
| SAM2 source | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| SAM checkpoint | SAM2.1 Hiera Small |
| Qwen service model | Qwen3.8-27B, BF16 |

[observed-packages.txt](../environment/observed-packages.txt) records packages installed in the simulator Python environment. It is an environment inventory, not a tested fresh-install lockfile; SAM2 and model serving have separate environments.

The RoboCasa checkout has a local change to `robocasa/models/objects/objects.py`. [robocasa-local.patch](../environment/robocasa-local.patch) records that change: generated object XML is placed in the episode temporary directory while relative asset references are resolved against the original source location and asset manifest. The checkout also contains a local shared-assets directory. Upstream assets, that shared asset data, model weights, and the driver rootfs must be provisioned separately. The robosuite checkout was clean when the release inventory was collected.

## Run the exact reported portfolio

Clone the repository on the existing host, then follow the commands in the [README](../README.md#reproduction). `prepare_checkout.py` creates an independent workflow directory. It rewrites only recipe paths, memory paths, and the reflection import path, leaving the source snapshots unchanged.

Use the generated `fullten-r1.json` with `evaluate.py` to select the exact ten cohort controllers. The generated `validated-routes.json` is intentionally different for Sink-to-Counter: it selects the reviewed successor, which independently succeeded with 642 actions after the cohort. The full cohort remains associated with the original controller and recordings.

The evaluator runs sequentially with seed 7 and 900 simulator actions per task. Modern routes have a 180-decision and 1,200-second episode budget. The outer launch timeout is 1,500 seconds. Legacy routes use their original `adaptive.skill_driver` entry and full argument lists. The current scripts are specifically scoped to the seed-7, 900-action protocol; editing manifest metadata alone does not change the hardcoded episode settings.

Do not reuse an existing output directory. The evaluator saves the initial manifest, memory copies, per-task launch commands and logs, native results, controller evidence, frames, and video. `summary.json` is updated as tasks finish, and `success_rate` is added only when all manifest tasks complete. An infrastructure exception can interrupt a cohort; an incomplete summary is not a completed SR measurement.

## Run learning and prepare another round

`run_workflow.py` executes:

1. `plan_from_memory.py` to retrieve recipes and same-task failure lessons.
2. `evaluate.py` to run the frozen cohort.
3. `learn_completed.py` to reflect on completed episodes and write `memory-after.json`.
4. `plan_from_memory.py` to write `next/manifest.json` and its retrieved-memory files.

The separate `next/` directory prevents next-round retrieval from overwriting the current round's inputs. Learned cards for known tasks retain the original successful memory snapshots. Experimental fallbacks receive the newly retrieved lesson context. No recurring monitor or scheduled automation is part of the workflow.

## Porting to another machine

A port needs the pinned simulator sources and patch, assets and asset manifest, SAM2 installation/checkpoint, compatible rendering environment, the Qwen service and its model-client configuration, and replacement of the launcher paths and account settings listed above. The path-preparation utility only relocates this repository's recipes. It does not perform that port. No fresh-machine native execution has been validated for this release.

## Validation performed for publication

The release validation checks the ten saved native outcomes against the summary, checks recipe arguments and memory relocation, verifies that all ten MP4 files decode, and checks the English documentation and gallery links. It also runs the memory ordering test already supplied with the workflow. These checks detect packaging errors; they are not new native robot episodes.

## English publication normalization

All repository documentation, source comments, filenames, and textual records are English. Four occurrences of a truncated directional phrase in the earlier Microwave-to-Counter baseline wrapper were translated to English; that file includes a publication note. Its recorded actions and outcomes are unchanged. The latest full ten-task cohort result files and controller snapshots are preserved without this translation.

## Video sharing

The public sharing entry point is [videos/README.md](../videos/README.md), with a poster, official outcome, and direct MP4 link for every task. `docs/index.html` is an optional HTML gallery source. GitHub Pages is not enabled: the publishing account has repository write permission, and the Pages creation request returned HTTP 404. An organization administrator can enable Pages from `main:/docs` later. The public repository and MP4 links are already available without Pages.
