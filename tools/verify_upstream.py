"""Compare default complete attacks with the pinned original checkout.

Run from the package folder: python tools/verify_upstream.py /path/to/upstream
The original checkout must contain env/. No models or training are used.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
upstream = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'PROVENANCE.json').read_text())
for relative, item in manifest['files'].items():
    assert hashlib.sha256((upstream / relative).read_bytes()).hexdigest() == item['original_sha256'], relative
sys.path.insert(0, str(upstream))
from env.siege_env import SiegeEnv as Original
from gymnasium_base_siege import BaseSiegeEnv
from gymnasium_base_siege.core import config as C

total_steps = 0
attacks = 0
for motif in C.MOTIFS:
    for seed in (0, 42, 100):
        for fixed in (False, True):
            a = Original(base_seed=seed, motifs=(motif,), fixed_army=fixed)
            b = BaseSiegeEnv(base_seed=seed, motifs=(motif,), fixed_army=fixed)
            oa, ia = a.reset(seed=17)
            ob, ib = b.reset(seed=17)
            rng = np.random.default_rng(seed)
            while True:
                for key in oa:
                    np.testing.assert_array_equal(oa[key], ob[key])
                assert ia == {k: v for k, v in ib.items() if k not in ('action_mask', 'settings_hash')}
                np.testing.assert_array_equal(a.action_masks(), b.action_masks())
                assert a.state.grid.topology_version == b.state.grid.topology_version
                # Compare every physical snapshot field from the original.
                sa, sb = a.snapshot(), b.snapshot()
                for k in ('action_mask', 'settings_hash'):
                    sb.pop(k)
                for building in sb['buildings']:
                    building.pop('range'); building.pop('min_range')
                assert sa == sb
                if a.state.is_finished or a.truncated:
                    break
                action = int(rng.choice(np.flatnonzero(a.action_masks())))
                oa, ra, ta, ca, ia = a.step(action)
                ob, rb, tb, cb, ib = b.step(action)
                assert (ra, ta, ca) == (rb, tb, cb)
                total_steps += 1
            attacks += 1
            a.close(); b.close()
print(json.dumps({'attacks': attacks, 'matching_steps': total_steps, 'result': 'identical defaults'}))
