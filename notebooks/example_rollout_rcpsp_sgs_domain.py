import logging

from discrete_optimization.rcpsp.rcpsp_parser import parse_file, get_data_available, RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from rcpsp_domains.rcpsp_sk_domain import RCPSPSGSDomain
from skdecide.utils import rollout, rollout_episode
logging.basicConfig(level=logging.INFO)


def rollout_rcpsp():
    print(get_data_available()[0])
    model: RCPSPModel = parse_file(get_data_available()[0])
    from discrete_optimization.rcpsp.rcpsp_solvers import CPSatRCPSPSolver, ParametersCP
    from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
    solver = CPSatRCPSPSolver(model)
    p = ParametersCP.default_cpsat()
    p.time_limit = 10
    res = solver.solve(parameters_cp=p, callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO,
                                                                   end_verbosity_level=logging.INFO)])
    sol, fit = res.get_best_solution_fit()
    print(model.evaluate(sol), model.satisfy(sol))
    domain_sk = RCPSPSGSDomain(model)
    for i in range(100):
        rollout(domain=domain_sk, from_memory=domain_sk.reset(),
                verbose=False)
        print("Final time : ", domain_sk.state[-1, 1])
        solution_rcpsp = RCPSPSolution(problem=model,
                                       rcpsp_schedule={t: {"start_time":
                                                           domain_sk.state[domain_sk.task_to_index[t], 1],
                                                           "end_time":
                                                           domain_sk.state[domain_sk.task_to_index[t], 1]+
                                                           domain_sk.dur[domain_sk.task_to_index[t]]}
                                                       for t in model.tasks_list})
        print(model.evaluate(solution_rcpsp), model.satisfy(solution_rcpsp))


if __name__ == "__main__":
    rollout_rcpsp()