import csv
import importlib.util
import json
from importlib.util import find_spec
import sys
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cpu_lowcond_proxy_pipeline(tmp_path):
    root = Path(__file__).resolve().parent / "cpu_lowcond_gmres_ir"
    sys.path.insert(0, str(root))
    try:
        runner = load_module(root / "run_cpu_lowcond_experiment.py", "run_cpu_lowcond_experiment")
        plotter = None
        if find_spec("matplotlib") is not None:
            plotter = load_module(root / "plot_cpu_lowcond_results.py", "plot_cpu_lowcond_results")

        data_dir = tmp_path / "data"
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        args = runner.build_arg_parser().parse_args(
            [
                "--mode",
                "proxy",
                "--train-count",
                "4",
                "--test-count",
                "3",
                "--size-min",
                "8",
                "--size-max",
                "10",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(results_dir),
            ]
        )
        runner.run_experiment(args)

        with (results_dir / "summary.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert {row["weight"] for row in rows} == {"W1", "W2"}
        assert {row["tolerance"] for row in rows} == {"1e-06", "1e-08"}
        for row in rows:
            config_dir = results_dir / row["config"]
            assert (config_dir / "train_curve.csv").exists()
            assert (config_dir / "test_results.csv").exists()
            assert (config_dir / "policy.json").exists()
            with (config_dir / "policy.json").open() as policy_handle:
                policy = json.load(policy_handle)
            assert policy["algorithm"] == "epsilon_greedy_contextual_bandit_q_table"
            assert policy["state_bins"] == {"condition": 10, "matrix_norm": 10}
            assert "state_bounds" in policy
            assert "q_table" in policy
            with (config_dir / "test_results.csv").open(newline="") as result_handle:
                result_rows = list(csv.DictReader(result_handle))
            assert {"selected_action", "uf", "ug", "u", "ur", "state_idx", "cond_bin", "norm_bin"}.issubset(result_rows[0])

        if plotter is not None:
            plotter.main(["--input-dir", str(results_dir), "--output-dir", str(figures_dir), "--font-size", "10"])
            assert (figures_dir / "cpu_lowcond_summary.png").exists()
            assert (figures_dir / "cpu_lowcond_training_curves.png").exists()
            assert (figures_dir / "cpu_lowcond_test_details.png").exists()
    finally:
        sys.path = [item for item in sys.path if item != str(root)]
