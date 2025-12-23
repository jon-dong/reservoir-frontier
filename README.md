# Reservoir Frontier

Code for analyzing information propagation frontiers in reservoir computing networks.

**Paper**: [Revisiting Deep Information Propagation: Fractal Frontier and Finite-size Effects](https://arxiv.org/abs/2508.03222)

## Installation

Requires Python >= 3.12

```bash
# With uv (recommended)
uv sync

# With pip
pip install -e .
```

## Project Structure

### Core Modules (`src/`)

- **`network.py`** - Reservoir network architectures (random, structured, convolutional)
- **`propagation.py`** - Information propagation tests across parameter grids
- **`fractal.py`** - Fractal dimension estimation via box-counting
- **`linop.py`** - Linear operator implementations
- **`utils.py`** - Utilities (plotting, linear regression)

### Scripts

- **`scripts/compute_frontier.py`** - Compute high-resolution frontiers (1000×1000 grid, zoomed parameter ranges)
- **`scripts/compute_lines.py`** - Compute temporal evolution data (100×100 grid, full parameter range)
- **`scripts/fractal_analysis.ipynb`** - Analyze fractal dimensions of experimental data
- **`scripts/plot_lines.ipynb`** - Visualize temporal evolution and fit decay rates

## Usage

### Compute Frontiers
```bash
# Edit parameters in compute_frontier.py (network config, resolution, bounds)
python scripts/compute_frontier.py
# Outputs: frontier plot + data to data/runs/
```

### Temporal Analysis
```bash
# Generate data
python scripts/compute_lines.py
# Analyze results
jupyter notebook scripts/plot_lines.ipynb
```

### Fractal Analysis
```bash
jupyter notebook scripts/fractal_analysis.ipynb
```

## Key Parameters

- `width`, `depth` - Network dimensions
- `mode` - Architecture: `'rand'`, `'struct'`, `'conv'`
- `W_scale_bounds`, `b_scale_bounds` - Parameter ranges to scan
- `resolution` - Grid resolution `[n_W, n_b]`
- `n_save_last` - Number of timesteps to save (1 for frontier, >1 for temporal)

## Citation

```bibtex
@article{dong2025fractal,
  title={Fractal Structure of Neural Network Loss Landscapes},
  author={Dong, Jianke and others},
  journal={arXiv preprint arXiv:2508.03222},
  year={2025}
}
```
