from run_pde_experiment import run_configs


if __name__ == "__main__":
    run_configs(
        default_configs=["2", "4"],
        description="Run W2 PDE experiments and write results_pde2/results_pde4.",
    )
