import os

import numpy as np
from matplotlib import pyplot as plt

os.environ["RAY_DEDUP_LOGS"] = "0"
import logging
from copy import deepcopy

from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_parser import (RCPSPModel,
                                                      get_data_available,
                                                      parse_file)
from rcpsp_sk_domain_local_search import (
    ParamsDomainEncodingLSOneStep, RCPSP_LS_Domain, RCPSP_LS_Domain_OneStep,
    records)
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.utils import rollout

logging.basicConfig(level=logging.INFO)


def rollout_rcpsp():
    print(get_data_available()[0])
    model: RCPSPModel = parse_file(get_data_available()[0])
    from discrete_optimization.generic_tools.callbacks.loggers import \
        ObjectiveLogger
    from discrete_optimization.rcpsp.rcpsp_solvers import (CPSatRCPSPSolver,
                                                           ParametersCP)

    solver = CPSatRCPSPSolver(model)
    p = ParametersCP.default_cpsat()
    p.time_limit = 10
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol, fit = res.get_best_solution_fit()
    print(model.evaluate(sol), model.satisfy(sol))
    domain_sk = RCPSP_LS_Domain_OneStep(model)
    for i in range(100):
        rollout(domain=domain_sk, from_memory=domain_sk.reset(), verbose=False)
        print("Final time : ", domain_sk.state[-1, 1])
        solution_rcpsp = RCPSPSolution(
            problem=model,
            rcpsp_schedule={
                t: {
                    "start_time": domain_sk.state[domain_sk.task_to_index[t], 1],
                    "end_time": domain_sk.state[domain_sk.task_to_index[t], 1]
                    + domain_sk.dur[domain_sk.task_to_index[t]],
                }
                for t in model.tasks_list
            },
        )
        print(model.evaluate(solution_rcpsp), model.satisfy(solution_rcpsp))


def solve_rcpsp_rllib():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    model: RCPSPModel = parse_file(file)
    domain_sk = RCPSP_LS_Domain_OneStep(model)
    from ray.rllib.algorithms.ppo import PPO

    config = PPO.get_default_config()
    config.num_rollout_workers = 5
    config.num_env_runners = 5
    solver = RayRLlib(
        domain_factory=lambda: domain_sk.shallow_copy(),
        config=config,
        algo_class=PPO,
        train_iterations=100,
    )
    solver.solve()
    # Start solving
    for k in range(100):
        rollout(domain=domain_sk, solver=solver, verbose=False)
        print("Final time : ", domain_sk.state[-1])
    solution_rcpsp = RCPSPSolution(
        problem=model,
        rcpsp_schedule={
            t: {
                "start_time": domain_sk.state[domain_sk.task_to_index[t]],
                "end_time": domain_sk.state[domain_sk.task_to_index[t]]
                + domain_sk.dur[domain_sk.task_to_index[t]],
            }
            for t in model.tasks_list
        },
    )
    print(model.evaluate(solution_rcpsp), model.satisfy(solution_rcpsp))


def solve_rcpsp_stable_baseline():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    model: RCPSPModel = parse_file(file)
    from discrete_optimization.generic_tools.callbacks.loggers import \
        ObjectiveLogger
    from discrete_optimization.rcpsp.rcpsp_solvers import (CPSatRCPSPSolver,
                                                           ParametersCP)

    solver = CPSatRCPSPSolver(model)
    p = ParametersCP.default_cpsat()
    p.time_limit = 2
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol, fit = res.get_best_solution_fit()
    print("Solution found by CP : ", model.evaluate(sol), model.satisfy(sol))
    print("Status solver : ", solver.get_status_solver())
    params = ParamsDomainEncodingLSOneStep(action_as_float=True, action_as_int=False)
    domain_sk = RCPSP_LS_Domain_OneStep(model, params_domain_encoding=params)
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC

    solver_args = {
        "baselines_policy": "MlpPolicy",
        "learn_config": {"total_timesteps": 300000},
        "verbose": 1,
        "learning_rate": 0.005,
        # "n_steps": 1000,
        # "ent_coef": 0.03,
        # "normalize_advantage": True,
        # "use_sde": True,
        # "sde_sample_freq":4,
        # "n_steps": 50,
        # "clip_range": 0.3,
        # "batch_size": 100,
        # "gae_lambda":1.0,
        # "ent_coef":0.01,
        # "vf_coef":0.5,
    }
    solver_args.update(
        {
            "policy_kwargs": dict(
                net_arch=[
                    dict(pi=[256, 256, 128, 128, 128], vf=[256, 256, 128, 128, 128])
                ]
            )
        }
    )
    solver_args["algo_class"] = A2C
    solver = StableBaseline(domain_factory=lambda: domain_sk, **solver_args)
    solver.solve()
    fig, ax = plt.subplots(1)
    records_ = np.array(records)
    ax.plot(np.convolve(records_, np.ones(30) / 30, mode="valid"))
    for k in range(100):
        episodes = rollout(
            domain=domain_sk,
            solver=solver,
            num_episodes=1,
            return_episodes=1,
            verbose=False,
        )
        print("Final time : ", domain_sk.state[-1])
        print("Actions = ", [str(x) for x in episodes[0][1]])
        solution_rcpsp = RCPSPSolution(
            problem=model,
            rcpsp_schedule={
                t: {
                    "start_time": domain_sk.state[domain_sk.task_to_index[t]],
                    "end_time": domain_sk.state[domain_sk.task_to_index[t]]
                    + domain_sk.dur[domain_sk.task_to_index[t]],
                }
                for t in model.tasks_list
            },
        )
        print(model.evaluate(solution_rcpsp), model.satisfy(solution_rcpsp))
    plt.show()


if __name__ == "__main__":
    solve_rcpsp_stable_baseline()
