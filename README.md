# Replicator discovery in BFF

This repository contains the code and data for *Replicator Discovery Is Faster Without Population Coupling in a Self-Modifying Program Soup*. It includes the BFF simulator, the experiment launchers, the compact data used in the paper, the analysis code, the paper figures, and a static companion website.

The three regimes used throughout the repository are:

| Regime | Description |
| --- | --- |
| U | Uniform sampling. Random programs are tested without execution. |
| N | Random-partner execution. A program persists while each interaction uses a fresh random partner. |
| R | Population-coupled execution. Both interacting programs come from, and return to, the evolving soup. |

The full raw runs are large, so the repository contains compact tables with every value needed to validate the reported results and rebuild the figures. The launchers for regenerating the raw runs are included as well.

## Quick start

You need Python 3.10 or later, a C++17 compiler, `make`, `pkg-config`, and the Brotli development libraries. On Debian or Ubuntu, the system dependencies can be installed with:

```sh
sudo apt-get install build-essential pkg-config libbrotli-dev python3-venv
```

Create a Python environment and run the lightweight checks:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt

make validate
make figures
make CUDA=0 smoke
```

These commands validate the committed paper data, rebuild the four PDFs in `figures/`, and run a three-epoch CPU comparison for U, N, and R. A GPU is not needed for any of them.

The main reported discovery results in the committed data are:

- U: 15 robust replicators in 209,715,200,000 samples, corresponding to an estimated median wait of 73,935.7 epochs;
- N: a median first-discovery epoch of 1,220, with discovery in all 100 runs; and
- R: a median first-discovery epoch of 2,453, with discovery in 99 of 100 runs.

## CUDA replay checks

The controlled replays for activation and pre-replicator availability require an NVIDIA GPU and the CUDA toolkit:

```sh
make CUDA=1 replays

python3 experiments/run_replays.py activation --profile smoke \
  --output runs/activation_smoke --gpus 0
python3 experiments/run_replays.py pre-replicator --profile smoke \
  --output runs/pre_replicator_smoke --gpus 0
python3 tests/validate_replay_smoke.py \
  --activation-root runs/activation_smoke \
  --pre-replicator-root runs/pre_replicator_smoke
```

Use a new output directory if you repeat these commands; the replay runner will not overwrite an existing run.

## Full paper runs

The full profile uses 131,072 programs, 16,000 epochs, and 100 seeds per regime. It is computationally expensive and must be enabled explicitly with `--confirm-full`.

Build the CUDA simulator, try one seed, and then schedule the full run for the GPUs available on your machine:

```sh
make CUDA=1

python3 experiments/run_regimes.py --profile paper --confirm-full \
  --regimes N,R --seeds 0 --output runs/paper_seed0 --gpus 0

python3 experiments/run_regimes.py --profile paper --confirm-full \
  --regimes U,N,R --seeds 0-99 --output runs/paper \
  --workers 4 --gpus 0,1,2,3
```

All launchers accept smaller seed, population, and epoch selections. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the commands used by each experiment.

## Companion website

The website has no runtime dependencies or backend:

```sh
python3 -m http.server 8000 --directory web
```

Open <http://localhost:8000> in a browser. To create the self-contained deployment directory used by GitHub Pages, run:

```sh
make website
python3 -m http.server 8000 --directory web/dist
```

## Repository layout

```text
src/          BFF simulator and CUDA replay programs
experiments/  smoke-test and full-experiment launchers
analysis/     data export, pathway analysis, and figure generation
data/         compact paper data, replay manifests, and reference outputs
figures/      paper figures generated from the committed data
tests/        validation scripts for data and smoke runs
web/          static companion website and browser-ready data
docs/         detailed reproduction notes
```

## Acknowledgments and license

The simulator in this repository is derived from the original [CuBFF implementation](https://github.com/paradigms-of-intelligence/cubff), published by the Paradigms of Intelligence project. We thank its authors for making the C++/CUDA code available. The derived files retain the original Google LLC copyright notices and are released under the Apache License 2.0.
