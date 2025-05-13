from pathlib import Path

import numpy as np
from corner import corner
from matplotlib import pyplot as plt

project_directory: Path = Path.cwd()
current_directory: Path = Path(__file__).parent

plt.style.use(project_directory / "arthur.mplstyle")

retrieval_run_name: str = "R1a_retrieval"

sampled_points_filepath: Path = current_directory / f"{retrieval_run_name}_points.npy"
log_likelihoods_filepath: Path = current_directory / f"{retrieval_run_name}_log_l.npy"
log_weights_filepath: Path = current_directory / f"{retrieval_run_name}_log_w.npy"
evidences_filepath: Path = current_directory / f"{retrieval_run_name}_log_z.npy"

sampled_points: np.ndarray = np.load(sampled_points_filepath)
log_likelihoods: np.ndarray = np.load(log_likelihoods_filepath)
log_weights: np.ndarray = np.load(log_weights_filepath)
evidences: np.ndarray = np.load(evidences_filepath)

plot_labels: list[str] = [
    "[CH$_4$]",
    "[H$_2$O]",
    r"$E_1$",
    r"$\log_{10}\!\left(E_0\right)$",
]

figure, axis = plt.subplots(len(plot_labels), len(plot_labels), figsize=(15, 15))

corner(
    sampled_points,
    weights=np.exp(log_weights),
    fig=figure,
    bins=20,
    labels=plot_labels,
    show_titles=True,
    title_fmt=".3f",
    title_kwargs=dict(fontsize=19),
    color="mediumseagreen",
    plot_datapoints=False,
    range=np.repeat(0.999, len(plot_labels)),
)

figure.savefig(
    current_directory / f"{retrieval_run_name}_corner_plot.pdf", bbox_inches="tight"
)
