from pathlib import Path

import numpy as np
from corner import corner
from matplotlib import pyplot as plt

project_directory: Path = Path.cwd()
current_directory: Path = Path(__file__).parent

plt.style.use(project_directory / "arthur.mplstyle")

retrieval_run_name: str = "2M2236b_G395H_logg-normal"
run_date_tag: str = "2025Jun15_19:50:01"
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
    r"$\log_{10}\!\left(g\right)$",
    "[H$_2$O]",
    "[CO]",
    "[CO$_2$]",
    "[CH$_4$]",
    "[Na+K]",
    "[H$_2$S]",
    "[NH$_3$]",
    r"$T_{\rm{phot.}}$",
    r"$T_{\rm{shallow 0}}$",
    r"$T_{\rm{shallow 1}}$",
    r"$T_{\rm{shallow 2}}$",
    r"$T_{\rm{shallow 3}}$",
    r"$T_{\rm{shallow 4}}$",
    r"$T_{\rm{deep 0}}$",
    r"$T_{\rm{deep 1}}$",
    r"$T_{\rm{deep 2}}$",
    r"$T_{\rm{deep 3}}$",
    r"$E_1$",
    r"$\log_{10}\!\left(E_0\right)$",
    r"$f_\mathrm{conv}$",
    r"$F_\mathrm{NRS1}$",
    r"$F_\mathrm{NRS2}$",
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
    color="cornflowerblue",
    plot_datapoints=False,
    range=np.repeat(0.999, len(plot_labels)),
)

figure.savefig(
    current_directory / f"{retrieval_run_prefix}_corner_plot.pdf", bbox_inches="tight"
)
