from pathlib import Path

import numpy as np
from corner import corner
from matplotlib import pyplot as plt

project_directory: Path = Path.cwd() / "potluck"
current_directory: Path = Path(__file__).parent

plt.style.use(project_directory / "arthur.mplstyle")

retrieval_run_name: str = "RISOTTO_T3B_Full"
run_date_tag: str = "2025-09-24"
retrieval_run_prefix: str = f"{retrieval_run_name}_{run_date_tag}"

sampled_points_filepath: Path = current_directory / f"{retrieval_run_prefix}_points.npy"
log_likelihoods_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_l.npy"
log_weights_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_w.npy"
evidences_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_z.npy"

sampled_points: np.ndarray = np.load(sampled_points_filepath)
log_likelihoods: np.ndarray = np.load(log_likelihoods_filepath)
log_weights: np.ndarray = np.load(log_weights_filepath)
evidences: np.ndarray = np.load(evidences_filepath)

plot_labels: list[str] = [
    r"R/R$_{\oplus}$",
    r"$T_{\rm{iso.}}$",
    r"$\log_{10}\!\left(P_{\rm{surf}}/\rm{bar}\right)$",
    "[He]",
    "[H$_2$O]",
    "[CH$_4$]",
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
    color="crimson",
    plot_datapoints=False,
    range=np.repeat(0.999, len(plot_labels)),
)

figure.savefig(
    current_directory / f"{retrieval_run_prefix}_corner_plot.pdf", bbox_inches="tight"
)
