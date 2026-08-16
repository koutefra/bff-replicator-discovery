# Reproducing the results

The repository provides two levels of reproduction:

- The committed compact data can be validated and used to rebuild every paper figure on a CPU.
- The raw experiments can be rerun with the paper settings. These runs are much more expensive and normally require NVIDIA GPUs.

Commands using `--profile smoke` are small checks of the complete execution path. Commands using `--profile paper` select the published configuration and require `--confirm-full` to prevent accidental long runs.

## Experiment map

| Result | Runner or analysis | Committed data |
| --- | --- | --- |
| Uniform baseline U | `experiments/run_regimes.py --regimes U` | `data/paper/uniform_replicators.csv` |
| First discovery in N and R | `experiments/run_regimes.py` | `data/paper/discovery_times.csv` |
| Discovery-event profile | `analysis/export_paper_data.py` | `data/paper/discovery_events.csv` |
| Operator composition | regime logs and `analysis/export_paper_data.py` | `data/paper/operator_trajectories.csv` |
| Discovery pathways | `analysis/extend_discovery_partner_reconstruction.py` | `data/pathways/` |
| Activation probability | `experiments/run_replays.py activation` | `data/paper/activation.csv` |
| Pre-replicator availability | `experiments/run_replays.py pre-replicator` | `data/paper/pre_replicator_availability.csv` |
| Long rewrite comparison | `experiments/run_rewrite_trace.py` | `data/paper/uniform_rewrite_summary.csv` |
| N-to-R perturbations | `experiments/run_perturbations.py` | generated logs, checkpoints, and PDFs |
| Paper figures | `analysis/make_figures.py` | `figures/` |

## Validate the release

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt

make validate
make figures
make website
```

`make validate` recomputes the N and R discovery medians from the committed per-run data, checks the replicator and event counts, the rewrite ratio, and that the website data matches the paper data. Other committed summaries are checked for presence and shape. `make figures` rebuilds all four paper PDFs. `make website` creates `web/dist/`.

To check the simulator itself on a CPU:

```sh
make CUDA=0 smoke
```

This runs three epochs with 128 programs in each regime and compares the scientific output with committed reference excerpts.

## U, N, and R discovery runs

The paper profile uses 131,072 programs of length 64, 16,000 analyzed epochs, no mutation, a replication-score threshold of 60, and seeds 0 through 99.

Start with the CPU smoke profile:

```sh
make CUDA=0
python3 experiments/run_regimes.py --profile smoke \
  --output runs/regime_smoke
python3 tests/validate_smoke.py --run-root runs/regime_smoke
```

For the full configuration, build with CUDA and choose the worker and GPU counts for your machine:

```sh
make CUDA=1
python3 experiments/run_regimes.py --profile paper --confirm-full \
  --regimes U,N,R --seeds 0-99 --output runs/paper \
  --workers 4 --gpus 0,1,2,3
```

The runner writes one log, stdout file, stderr file, and metadata file per regime and seed. It records epochs 1 through 16,000. Existing results are never overwritten; pass `--resume` to skip runs that already have a non-empty log.

The compact U/N/R tables in this release were exported from the original raw-results layout. To repeat that export, place the logs under the following paths inside a raw-results root:

```text
runs/reinit/log_*.log
runs/no_mutation/random/log_*.log
runs/no_mutation/interaction/log_*.log
```

Then run:

```sh
python3 analysis/export_paper_data.py \
  --raw-root /path/to/raw-results-root \
  --output data/paper \
  --operator-stride 100
```

When working from new `run_regimes.py` output, the `u`, `n`, and `r` log directories can be linked or copied into the layout above before exporting.

## Discovery pathways

Recompute the summary of the 100 committed pathway reconstructions with:

```sh
python3 analysis/summarize_pathways.py
```

The table contains 50 N events and 50 R events. Every event has an exact replay, and the dominant pathway occurs in 43 N events and 45 R events.

Reconstructing those events from raw runs is expensive because the simulator must recreate selected checkpoints. Given the full N and R log directories, use:

```sh
python3 analysis/extend_discovery_partner_reconstruction.py \
  --random-logs-dir /path/to/random \
  --interaction-logs-dir /path/to/interaction \
  --bin-main bin/main \
  --gpus 0,1,2,3 \
  --dumps-root runs/pathway_reconstruction \
  --analysis-out-dir runs/pathway_analysis
```

`analysis/build_discovery_prepost_table.py` can then materialize the corresponding before-and-after tapes.

## Activation and pre-replicator availability

These controlled replays use dedicated CUDA executables. The smoke profile runs one seed, three epochs, 128 programs, and a small subset of the fixed tapes:

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

The paper profiles are:

```sh
python3 experiments/run_replays.py activation --profile paper --confirm-full \
  --output runs/activation --workers 4 --gpus 0,1,2,3
python3 experiments/run_replays.py pre-replicator --profile paper --confirm-full \
  --output runs/pre_replicator --workers 4 --gpus 0,1,2,3
```

Activation uses seeds 0 through 3. Pre-replicator availability uses seeds `0,2,3,5,6,22,31,53` and 70 fixed focal tapes. Each runner executes both N and R and writes an aggregate summary under its output directory.

## Long rewrite comparison

```sh
python3 experiments/run_rewrite_trace.py --profile smoke \
  --output runs/rewrite_smoke
python3 experiments/run_rewrite_trace.py --profile paper --confirm-full \
  --output runs/rewrite_paper
```

The paper profile follows 100 programs for 16,000 matched transitions in N and R. The analysis counts an interaction when a contiguous block of at least 30 changed cells is rewritten to a single byte value.

## N-to-R perturbations

```sh
python3 experiments/run_perturbations.py --profile smoke \
  --output runs/perturb_smoke
python3 experiments/run_perturbations.py --profile paper --confirm-full \
  --output runs/perturb_paper
```

The paper profile begins with N seeds 0 through 3. It saves checkpoints at epochs 500, 1,000, 2,000, 4,000, and 8,000, continues a separate R branch from each checkpoint through epoch 16,000, and produces one operator-trajectory PDF per source seed.

## Figure conventions

The figures use a score threshold of 60 for discovery, a centered 500-epoch smoother for pre-replicator availability, a time-independent N pre-replicator baseline, and the approximation `30 × P(t) × A(t)`. R discovery events are exposure-normalized after takeover conditioning. Operator frequencies are reported as percentages of all program bytes.

Run the figure and data checks directly with:

```sh
python3 analysis/make_figures.py
python3 tests/validate_paper_data.py
```
