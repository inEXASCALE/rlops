import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module():
    module_path = Path(__file__).resolve().parent / "plot_training_curves.py"
    spec = importlib.util.spec_from_file_location("plot_training_curves", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_curves(root: Path, template: str) -> None:
    for config_id in range(1, 5):
        results_dir = root / template.format(config_id=config_id)
        results_dir.mkdir(parents=True)
        np.save(results_dir / "episode_rewards.npy", np.linspace(config_id, config_id + 1, 5))
        np.save(results_dir / "training_losses.npy", np.linspace(1 / config_id, 0.0, 5))


def test_combined_training_curve_outputs(tmp_path):
    module = load_module()
    module.apply_plot_style(10)

    for name, template in (
        ("dense", "results_random_dense{config_id}"),
        ("sparse", "results_random_sparse{config_id}"),
        ("pde", "results_pde{config_id}"),
    ):
        root = tmp_path / name
        output_dir = tmp_path / f"figures_{name}"
        write_curves(root, template)
        dataset = module.TrainingDataset(
            name=name,
            root=root,
            result_template=template,
            output_dir=output_dir,
            output_stem=f"{name}_training_curves",
            title_prefix=name.upper(),
        )

        assert module.plot_dataset(dataset)
        assert (output_dir / f"{name}_training_curves.pdf").exists()
        assert (output_dir / f"{name}_training_curves.png").exists()


def test_pde_training_label_fallbacks():
    module_path = Path(__file__).resolve().parent / "pde" / "visualize_pde_results.py"
    spec = importlib.util.spec_from_file_location("visualize_pde_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.format_training_label(Path("results_pde1")) == r"$W_1$, $\tau=10^{-6}$"
    assert module.format_training_label(Path("results_pde4")) == r"$W_2$, $\tau=10^{-8}$"
