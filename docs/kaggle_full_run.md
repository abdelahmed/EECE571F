# Kaggle Full Project Run

Use these notebook cells on Kaggle.

## Cell 1 — Clone or update repo

```python
!rm -rf /kaggle/working/EECE571F
!git clone https://github.com/abdelahmed/EECE571F.git
%cd /kaggle/working/EECE571F
```

## Cell 2 — Install dependencies

```python
!python -m pip install --upgrade pip
!python -m pip install -e .
!python -m pip install torch torchvision scipy imageio tqdm wandb matplotlib
```

## Cell 3 — Verify GPU

```python
import torch
print('cuda_available =', torch.cuda.is_available())
print('device_count =', torch.cuda.device_count())
print('device_name =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
```

## Cell 4 — Run all four experiments sequentially

```python
%cd /kaggle/working/EECE571F
!PYTHONPATH=src python scripts/phase9_run_four_experiments.py
```

## Cell 5 — Inspect outputs

```python
!ls -R /kaggle/working/EECE571F/artifacts/phase9
```

Expected files include:
- `kaggle_baseline_training_summary.json`
- `kaggle_saans_training_summary.json`
- `kaggle_saans_no_weighting_summary.json`
- `kaggle_saans_alpha05_summary.json`
- `phase9_comparison_report.md`
```