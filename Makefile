PYTHON=python3

.PHONY: demo test all-smokes phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase8-baseline phase8-saans phase8-compare phase9-saans-train phase9-no-weight phase9-alpha phase9-all

demo:
	PYTHONPATH=src $(PYTHON) scripts/smoke_demo.py

phase2:
	PYTHONPATH=src $(PYTHON) scripts/phase2_demo.py

phase3:
	PYTHONPATH=src $(PYTHON) scripts/phase3_prepare_qm9.py

phase4:
	PYTHONPATH=src $(PYTHON) scripts/phase4_baseline_smoke.py

phase5:
	PYTHONPATH=src $(PYTHON) scripts/phase5_timestep_diagnostics.py

phase6:
	PYTHONPATH=src $(PYTHON) scripts/phase6_saans_smoke.py

phase7:
	PYTHONPATH=src $(PYTHON) scripts/phase7_toy_study.py

phase8:
	PYTHONPATH=src $(PYTHON) scripts/phase8_experiment_matrix.py

phase8-baseline:
	PYTHONPATH=src $(PYTHON) scripts/phase8_run_baseline_short.py

phase8-saans:
	PYTHONPATH=src $(PYTHON) scripts/phase8_run_saans_short.py

phase8-compare:
	PYTHONPATH=src $(PYTHON) scripts/phase8_compare_short_runs.py

phase9-saans-train:
	PYTHONPATH=src $(PYTHON) scripts/phase9_run_saans_training.py

phase9-no-weight:
	PYTHONPATH=src $(PYTHON) scripts/phase9_run_no_weighting_training.py

phase9-alpha:
	PYTHONPATH=src $(PYTHON) scripts/phase9_run_alpha_ablation_training.py

phase9-all:
	PYTHONPATH=src $(PYTHON) scripts/phase9_run_four_experiments.py

all-smokes:
	PYTHONPATH=src $(PYTHON) scripts/run_all_smokes.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
