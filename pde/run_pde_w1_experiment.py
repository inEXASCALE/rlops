from run_pde_experiment import run_configs


if __name__ == "__main__":
    run_configs(
        default_configs=["1", "3"],
        description="Run W1 PDE experiments and write results_pde1/results_pde3.",
    )
