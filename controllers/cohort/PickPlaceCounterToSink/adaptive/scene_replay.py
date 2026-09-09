"""Simulator-only scene pinning and contact evidence for paired development trials."""
import json
from pathlib import Path
import numpy as np

SCENES = Path('/home/jli/state/fullten-current-20260908/scenes')

def reset_pinned_scene(environment, task, seed, run):
    env = environment.unwrapped
    scene = SCENES / f'{task}-seed{seed}.json'
    SCENES.mkdir(parents=True, exist_ok=True)
    # Initialize Gym's outer OrderEnforcing wrapper on replay as well.
    environment.reset()
    if not scene.exists():
        snapshot = {'model': env.sim.model.get_xml(),
                    'states': env.sim.get_state().flatten().tolist(),
                    'ep_meta': env.get_ep_meta()}
        scene.write_text(json.dumps(snapshot, default=lambda x: x.tolist()))
    snapshot = json.loads(scene.read_text())
    # Match RoboCasa's official dataset reset_to sequence, including task refs.
    env.set_ep_meta(snapshot['ep_meta'])
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(snapshot['model']))
    env.sim.reset()
    env.sim.set_state_from_flattened(np.asarray(snapshot['states']))
    env.sim.forward()
    if hasattr(env, 'update_sites'):
        env.update_sites()
    if hasattr(env, 'update_state'):
        env.update_state()
    actual = env.sim.get_state().flatten()
    if not np.array_equal(actual, np.asarray(snapshot['states'])):
        raise RuntimeError('Pinned initial simulator state did not restore exactly')
    (run / 'scene-replay.json').write_text(json.dumps({
        'scene_path': str(scene), 'task': task, 'seed': seed,
        'state_restored_exactly': True, 'ep_meta': env.get_ep_meta()},
        default=lambda x: x.tolist(), indent=2))
    return env.get_observation(env._get_observations(force_update=True))

def record_contacts(environment, run, sequence):
    sim = environment.unwrapped.sim
    contacts = []
    for contact in sim.data.contact[:sim.data.ncon]:
        contacts.append({'geom1': sim.model.geom_id2name(contact.geom1),
                         'geom2': sim.model.geom_id2name(contact.geom2),
                         'distance_m': float(contact.dist)})
    with (run / 'diagnostic-contacts.jsonl').open('a') as f:
        f.write(json.dumps({'sequence': sequence, 'contacts': contacts}) + '\n')
