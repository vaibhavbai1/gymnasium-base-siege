# Validation record — 26 October 2026

Release candidate: 0.1.0rc1. These are software checks, not training results or
an evaluation of learned policies. No model was trained.

## Tested dependency sets

Python 3.12.14 on Linux:

| Gymnasium | NumPy | Matplotlib | Result |
|---|---|---|---|
| 1.4.0 | 2.5.4 | 3.11.2 | 182 tests passed |
| 1.3.0 | 2.5.2 | 3.11.1 | 182 tests passed |

The suite includes six adapted original unit-test files plus public API and
settings tests. It covers combat, entities, generation, pathfinding, rendering,
Gymnasium registration, seeded repeatability, legal masks, observation bounds,
reward/cost accounting, terminal conditions, truncation, serialized settings,
instance isolation, vector reset/step and pickling.

Customization tests check actual damage and firing intervals, troop movement
credits, path costs, HP, purchase masks, defence coverage, render snapshot ranges,
invalid settings and zero-damage splash. Two differently configured environments
can coexist without changing global defaults.

## Original-default parity

Against the pinned upstream commit, **24 complete attacks / 507 decisions** matched
in both dependency sets. The check covers all four motifs, three layout seeds,
and both ordinary shopping and fixed-army mode, using identical legal actions.
Observations, rewards, masks, terminal flags, original info fields and physical
snapshots matched exactly at each comparison point. New public metadata fields
were excluded from this comparison.

This is strong regression evidence, not an exhaustive proof for every possible
layout/action sequence. Run `python tools/verify_upstream.py /path/to/upstream`
from this package folder to repeat it against a checkout of the pinned commit.
The tool first checks all ten original source hashes.

## Gymnasium checker scope

The public package keeps Gymnasium's passive checks enabled. The standalone
`check_env` assumes that unmasked random actions are executable. Base Siege instead
raises on illegal actions, so an unmodified check raises `ValueError` (observed:
`Illegal action: 209`). This limitation is documented, not hidden by changing
invalid actions into waits or by disabling passive checks.

The tests run `check_env` with a **test-only masked sampler**, and rendering is
tested separately. This checks seeded reset and step behavior while respecting
the task's action contract. It is not a claim of an unmodified checker pass.

The original environment overwrote Gymnasium's `spec` attribute with a base
layout. The public package uses `base_spec` for the layout instead; a regression
test checks that registration metadata survives reset.

## Remaining release work

MIT and the independent repository have been selected. Authenticate the upload
to TestPyPI, verify a fresh download, then publish to PyPI. Windows and macOS
have not been tested here. No claim of Farama acceptance or public availability
is made. All software tests used short synthetic or random attacks, not a new
capstone experiment.
