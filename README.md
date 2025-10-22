# ResView

ResView is a Reciprocal-Space Viewer (ResView) and RSM workflow GUI.

## Install
**pip**
```bash
pip install resview
resview --version
```

**Run**
```bash
resview
```

## Dev quickstart
```bash
python -m pip install -U pip build
python -m pip install -e .  # editable dev
python -m build             # sdist + wheel
```

## Environment variable
Set a defaults YAML for your experiment setup:
```bash
export RSM3D_DEFAULTS_YAML=/path/to/rsm3d_defaults.yaml
```
