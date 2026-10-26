# Publishing Base Siege

Gymnasium does not provide a package-upload service. Publish a Python package on
PyPI; importing it registers the environment with Gymnasium. A listing in the
Gymnasium documentation is a separate, reviewed request.

## 1. Finish release metadata

The source repository had no license file at the inspected commit. The owners
must choose the license before release. After that decision, add its full text
as LICENSE, set the SPDX `license` value and `license-files = ["LICENSE"]` under
`[project]` in pyproject.toml, and update the release status. Do not treat this
draft as already licensed. The public package author is Vaibhav Bairathi.

Choose a separate public GitHub repository, for example
`Reinforcement-Learning-UTS/gymnasium-base-siege`, and update package project URLs
to the actual repository. This is a proposed location, not an existing release.
Keep the original project link as provenance.

## 2. Install and verify locally

Extract the provided ZIP. Open a terminal **inside its gymnasium-base-siege folder**,
where pyproject.toml is located. Use Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[render,dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python examples/random_agent.py
.venv\Scripts\python examples/custom_settings.py
.venv\Scripts\python -m build
.venv\Scripts\python -m twine check dist/*
```

On macOS/Linux, in that same package folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[render,dev]"
.venv/bin/python -m pytest -q
.venv/bin/python examples/random_agent.py
.venv/bin/python examples/custom_settings.py
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

The conservative dependency bounds reflect the tested Python and library versions.
Broaden them only after testing more versions. The wheel itself is pure Python.

## 3. Publish the separate source repository

Create an empty repository through GitHub. From the same package folder run
`git init`, `git add .`, and `git commit -m "Prepare Base Siege Gymnasium package"`.
Follow the new repository's displayed commands to set its actual remote and push.
The included .gitignore excludes virtual environments and build output.
Do not use these commands in the original capstone repository.

## 4. Test the upload on TestPyPI

Create a TestPyPI account at https://test.pypi.org/ and configure its authentication.
TestPyPI and PyPI are separate services. Use a token in your own terminal; never
paste tokens into a chat or commit them to Git. In the package folder, using the
same virtual-environment Python executable from step 2:

```bash
python -m twine upload --repository testpypi dist/*
```

Here and below `python` means that virtual-environment executable; use its explicit
Windows or Unix path from step 2 if the virtual environment is not activated.
For token authentication the username is `__token__`; enter the token at the
password prompt. A GitHub trusted publisher is an alternative to storing tokens.

Test the uploaded wheel in a fresh virtual environment. Install its dependencies
from regular PyPI first, then fetch only this package from TestPyPI with
`--no-deps`. This avoids relying on TestPyPI for dependencies:

```bash
python -m pip install "gymnasium>=1.3,<1.5" "numpy>=2.5.2,<2.6"
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps gymnasium-base-siege==0.1.0rc1
python -c "import gymnasium as g; import gymnasium_base_siege; e=g.make('BaseSiege-v0'); print(e.reset(seed=0)[1]['motif'])"
```

## 5. Publish the public release

After the test upload works, update the version in pyproject.toml and the package
__init__.py to `0.1.0`, update README/release status, clear old generated dist/build
output, rerun tests and rebuild. Use your PyPI account at https://pypi.org/.
From the package folder:

```bash
python -m twine check dist/*
python -m twine upload dist/*
```

Only upload the intended release files. PyPI does not allow replacing an already
uploaded distribution with different bytes; publish a new version for fixes.
Verify a fresh installation from PyPI before announcing availability. Only then
advertise `pip install gymnasium-base-siege` as a working installation command.

## 6. Request an external-environment listing

Gymnasium's current external-environment page asks contributors to reach out on
Farama Discord and then submit a pull request editing its listing file. Follow
the current instructions there. Listing acceptance is not automatic and is not
required for `gym.make` to work after package installation/import.

Suggested short listing text after publication:

> Base Siege is a single-agent strategy environment with resource-constrained army
> selection and spatial deployment on a defended 16×16 grid. It provides legal-action
> masks, customizable troop and defence statistics, procedural layouts and RGB rendering.

Official references, checked 26 October 2026:
- https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
- https://gymnasium.farama.org/environments/third_party_environments/
- https://packaging.python.org/en/latest/tutorials/packaging-projects/
