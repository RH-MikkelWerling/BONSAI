.PHONY: test smoke readiness rarity-demo hierarchical-rarity-assemble

test:
	python -m pytest tests

smoke:
	python -m compileall bonsai opera tests
	python -c "import bonsai, opera; print('package imports ok')"

readiness:
	python -m opera.run.check_readiness --config opera/configs/sweep_example.yaml

rarity-demo:
	python -m opera.run.aggregate_results --results_dir ./results --output_dir ./results/aggregated --baseline tabular_ehr --rarity_plots

hierarchical-rarity-assemble:
	python -m opera.run.hierarchical_rarity --mode assemble
