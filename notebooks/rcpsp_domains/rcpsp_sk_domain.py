from typing import Optional

import networkx as nx
from discrete_optimization.rcpsp.rcpsp_utils import compute_graph_rcpsp

import skdecide
from skdecide import Space, Value, TransitionOutcome
from skdecide.builders.domain import Goals
from skdecide.hub.space.gym import GymSpace, DiscreteSpace, BoxSpace, MultiDiscreteSpace
from skdecide.domains import DeterministicPlanningDomain, RLDomain
import discrete_optimization
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
import numpy as np
import logging

logger = logging.getLogger(__name__)


class D(RLDomain):
    T_state = np.ndarray  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class RCPSPSGSDomain(D):
    def __init__(self, problem: RCPSPModel):
        self.problem = problem
        self.nb_tasks = self.problem.n_jobs
        self.nb_resources = len(self.problem.resources_list)
        self.dur = np.array(
            [self.problem.mode_details[t][1]["duration"]
             for t in self.problem.tasks_list],
            dtype=int,
        )
        self.resource_consumption = np.array(
            [[self.problem.mode_details[t][1].get(r, 0)
             for r in self.problem.resources_list]
             for t in self.problem.tasks_list],
            dtype=int,
        )
        self.rcpsp_successors = {t: set(s) for t, s in self.problem.successors.items()}
        self.graph = compute_graph_rcpsp(self.problem)
        self.graph_nx = self.graph.to_networkx()
        self.topological_order = list(nx.topological_sort(self.graph_nx))
        self.all_ancestors = self.graph.full_predecessors
        self.all_ancestors_order = {t: sorted(self.all_ancestors[t],
                                              key=lambda x: self.topological_order.index(x))
                                    for t in self.all_ancestors}
        self.task_to_index = {self.problem.tasks_list[i]: i
                              for i in range(self.problem.n_jobs)}
        self.index_to_task = {i: self.problem.tasks_list[i]
                              for i in range(self.problem.n_jobs)}
        self.initial_state = np.zeros(
            [self.nb_tasks, 2], dtype=np.int64
        )
        self.initial_resource_availability = np.array([self.problem.get_resource_availability_array(r)
                                                      for r in self.problem.resources_list],
                                                      dtype=int)
        self.state = np.copy(self.initial_state)
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0

    def _state_step(self, action: D.T_event) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        if self.state[action, 0]:
            return TransitionOutcome(self.state,
                                     Value(cost=0),
                                     termination=False,
                                     info=None)
        else:
            pre = self.cur_makespan
            tasks = [self.task_to_index[j] for j in self.all_ancestors_order[self.index_to_task[action]]
                     if self.state[self.task_to_index[j], 0] == 0] + [action]
            for k in tasks:
                self.insert_task(k)
            return TransitionOutcome(self.state,
                                     Value(cost=self.cur_makespan-pre),
                                     termination=np.all(self.state[:, 0] == 1),
                                     info=None)

    def insert_task(self, k: int):
        res_consumption = self.resource_consumption[k, :]
        min_date = self.state[k, 1]
        if self.dur[k] == 0:
            next_date = min_date
        else:
            next_date = next((t for t in range(min_date, 2*self.problem.horizon)
                              if all(np.min(self.resource_availability[p, t:t+self.dur[k]])>=res_consumption[p]
                                     for p in range(self.resource_availability.shape[0]))), None)
        self.state[k, 1] = next_date
        self.state[k, 0] = 1
        for t in range(next_date, next_date+self.dur[k]):
            self.resource_availability[:, t] -= res_consumption
        for succ in self.rcpsp_successors[self.index_to_task[k]]:
            self.state[self.task_to_index[succ], 1] = max(self.state[self.task_to_index[succ], 1],
                                                          next_date+self.dur[k])
        self.cur_makespan = max(self.cur_makespan, next_date+self.dur[k])

    def _get_action_space_(self) -> Space[D.T_event]:
        return DiscreteSpace(self.nb_tasks)

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        return DiscreteSpace(self.nb_tasks)

    def _state_reset(self) -> D.T_state:
        self.state = np.copy(self.initial_state)
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0
        return self.state

    def _get_observation(self, state: D.T_state, action: Optional[D.T_event] = None) -> D.T_observation:
        return state

    def _get_observation_space_(self) -> Space[D.T_observation]:
        return MultiDiscreteSpace(nvec=np.array([[1, self.problem.horizon]
                                                 for t in self.problem.tasks_list]))


