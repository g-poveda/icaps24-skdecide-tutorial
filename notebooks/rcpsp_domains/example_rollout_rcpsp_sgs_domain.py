import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import logging
import matplotlib.pyplot as plt
import numpy as np
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_parser import (RCPSPModel,
                                                      get_data_available,
                                                      parse_file)
from rcpsp_sk_domain import ParamsDomainEncoding, RCPSPSGSDomain
from stochastic_rcpsp_sk_domain import StochasticRCPSPSGSDomain, records
from copy import deepcopy
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
    domain_sk = RCPSPSGSDomain(model)
    for i in range(100):
        rollout(domain=domain_sk, verbose=False)  # from_memory=domain_sk.reset(),
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
    # file = [f for f in get_data_available() if "j1201_5.sm" in f][0]
    model: RCPSPModel = parse_file(file)
    domain_sk = RCPSPSGSDomain(
        model,
        params_domain_encoding=ParamsDomainEncoding(
            return_times_in_state=False,
            return_scheduled_in_state=True,
            use_cpm_for_cost=True,
            terminate_when_already_schedule=False,
            dummy_cost_when_already_schedule=30,
            use_additive_makespan_for_cost=False,
            nb_min_task_inserted=None,
            nb_max_task_inserted=25,
            filter_tasks=False,
            only_available_tasks=False,
        ),
    )
    from ray.rllib.algorithms.appo import APPO
    from ray.rllib.algorithms.bc import BC
    from ray.rllib.algorithms.dqn import DQN
    from ray.rllib.algorithms.impala import Impala
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.algorithms.sac import SAC
    ac = DQN.get_default_config()
    ac.lr = 5e-3
    # ac.framework("tf")
    # ac.num_env_runners = 8
    # ac.num_workers = 1
    # ac.num_env_runners = 1
    # ac.sgd_minibatch_size = 16
    # ac.rollout_fragment_length = "auto"
    # ac.log_level = "DEBUG"
    solver_factory = lambda: RayRLlib(
        domain_factory=lambda: deepcopy(domain_sk),
        algo_class=DQN,
        config=ac,
        train_iterations=100,
    )
    assert RayRLlib.check_domain(domain_sk)
    # Start solving
    with solver_factory() as solver:
        solver.solve()
        for i in range(20):
            domain_sk.reset()
            rollout(domain=domain_sk, solver=solver, verbose=False)
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


def solve_rcpsp_stable_baseline():
    file = [f for f in get_data_available() if "j1201_5.sm" in f][0]
    file = [f for f in get_data_available() if "j601_9.sm" in f][0]
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    model: RCPSPModel = parse_file(file)
    from discrete_optimization.generic_tools.callbacks.loggers import \
        ObjectiveLogger
    from discrete_optimization.rcpsp.rcpsp_solvers import (CPSatRCPSPSolver,
                                                           ParametersCP)
    from rcpsp_sk_domain import records
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
    domain_sk = RCPSPSGSDomain(
        model,
        params_domain_encoding=ParamsDomainEncoding(
            return_times_in_state=True,
            return_scheduled_in_state=True,
            use_cpm_for_cost=True,
            terminate_when_already_schedule=False,
            dummy_cost_when_already_schedule=1,
            use_additive_makespan_for_cost=False,
            nb_min_task_inserted=1,
            nb_max_task_inserted=None,
            filter_tasks=False,
            only_available_tasks=False,
        ),
    )
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC

    solver_args = {
        "baselines_policy": "MlpPolicy",
        "learn_config": {"total_timesteps": 300000},
        "verbose": 1,
        # "learning_rate": 0.0001,
        "n_steps": 300,
        # "batch_size": 100
    }
    solver_args.update(
        {
            "policy_kwargs": dict(
                net_arch=[dict(pi=[256, 256, 128, 128], vf=[256, 256, 128, 128])]
            )
        }
    )
    solver_args["algo_class"] = A2C
    solver = StableBaseline(domain_factory=lambda: domain_sk, **solver_args)
    # Start solving
    solver.solve()

    for k in range(100):
        episodes = rollout(
            domain=domain_sk,
            solver=solver,
            num_episodes=1,
            verbose=False,
            return_episodes=True,
        )
        print(len(episodes[0][0]))
        print(episodes[0][1])
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
    fig, ax = plt.subplots(1)
    records = np.array(records)
    ax.plot(np.convolve(records, np.ones(30) / 30, mode="valid"))
    plt.show()


def solve_stochastic_rcpsp_stable_baseline():
    file = [f for f in get_data_available() if "j301_8.sm" in f][0]
    file = [f for f in get_data_available() if "j601_9.sm" in f][0]
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
    domain_sk = StochasticRCPSPSGSDomain(
        model,
        params_domain_encoding=ParamsDomainEncoding(
            return_times_in_state=True,
            return_scheduled_in_state=True,
            use_cpm_for_cost=True,
            terminate_when_already_schedule=False,
            dummy_cost_when_already_schedule=20,
            use_additive_makespan_for_cost=True,
            nb_min_task_inserted=8,
            nb_max_task_inserted=20,
            only_available_tasks=False,
        ),
    )
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC

    solver_args = {
        "baselines_policy": "MlpPolicy",
        "learn_config": {"total_timesteps": 300000},
        "verbose": 1,
        # "learning_rate": 0.05,
        "n_steps": 300,
        # "batch_size": 100
    }
    solver_args.update(
        {
            "policy_kwargs": dict(
                net_arch=[dict(pi=[256, 256, 256, 128], vf=[256, 256, 256, 128])]
            )
        }
    )
    solver_args["algo_class"] = A2C
    solver = StableBaseline(domain_factory=lambda: domain_sk, **solver_args)
    # Start solving
    solver.solve()
    fig, ax = plt.subplots(1)
    records_ = np.array(records)
    ax.plot(np.convolve(records_, np.ones(30) / 30, mode="valid"))
    for k in range(100):
        episodes = rollout(
            domain=domain_sk,
            solver=solver,
            num_episodes=1,
            verbose=False,
            return_episodes=True,
        )
        print(len(episodes[0][0]))
        print(episodes[0][1])
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
    plt.show()


if __name__ == "__main__":
    solve_rcpsp_rllib()
