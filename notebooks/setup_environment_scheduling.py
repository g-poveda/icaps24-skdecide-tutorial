import importlib
import importlib.metadata
import os
import subprocess


def install_additional_dependencies(force_reinstall: bool = False):
    if (
        "optuna" in [x.name for x in importlib.metadata.distributions()]
        and not force_reinstall
    ):
        print(
            "Optuna is already installed and we are asked not to forcibly reinstall it"
        )
    else:
        subprocess.run(f"pip --default-timeout=1000 install optuna", shell=True)
    if (
        "optuna-dashboard" in [x.name for x in importlib.metadata.distributions()]
        and not force_reinstall
    ):
        print(
            "Optuna-Dashboard is already installed and we are asked not to forcibly reinstall it"
        )
    else:
        subprocess.run(
            f"pip --default-timeout=1000 install optuna-dashboard", shell=True
        )
    if (
        "dash" in [x.name for x in importlib.metadata.distributions()]
        and not force_reinstall
    ):
        print("Dash is already installed and we are asked not to forcibly reinstall it")
    else:
        subprocess.run(f"pip --default-timeout=1000 install dash", shell=True)


def download_files_needed():
    this_folder = os.path.abspath(os.path.dirname(__file__))
    module_path = os.path.join(this_folder, "rcpsp_domains/")
    if not os.path.exists(module_path):
        os.makedirs(module_path)
    for script in [
        "rcpsp_sk_domain.py",
        "rcpsp_sk_domain_local_search.py",
        "stochastic_rcpsp_sk_domain.py",
        "multi_solve_optuna.py",
        "optuna_journal_offline.log"
    ]:
        subprocess.run(
            f"wget https://raw.githubusercontent.com/fteicht/icaps24-skdecide-tutorial/gpd/rcpsp_rl/notebooks/rcpsp_domains/{script} -O {os.path.join(module_path, script)}",
            shell=True,
        )
