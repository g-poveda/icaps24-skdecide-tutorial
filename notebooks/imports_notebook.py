import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from typing import Any, Dict, List, Optional, Set, Union
import matplotlib.pyplot as plt
# [allow for running minizinc inside a notebook]
import nest_asyncio
import numpy as np
import seaborn as sns
from discrete_optimization.datasets import fetch_data_from_psplib
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_parser import (get_data_available,
                                                      parse_file)
from discrete_optimization.rcpsp.rcpsp_utils import (plot_ressource_view,
                                                     plot_task_gantt)
from skdecide import rollout
from skdecide.builders.domain.scheduling.modes import (ConstantModeConsumption,
                                                       ModeConsumption)
from skdecide.builders.domain.scheduling.scheduling_domains import (
    SchedulingObjectiveEnum, SingleModeRCPSP)
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.solver.do_solver.do_solver_scheduling import (DOSolver,
                                                                SolvingMethod)
from skdecide.hub.solver.do_solver.sgs_policies import (BasePolicyMethod,
                                                        PolicyMethodParams,
                                                        PolicyRCPSP)
from skdecide.hub.solver.do_solver.sk_to_do_binding import \
    from_last_state_to_solution
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.stable_baselines import StableBaseline

nest_asyncio.apply()
