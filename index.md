---
title: "ICAPS 2024 Tutorial on Scikit-Decide"
subtitle: "A Hands-On Tutorial on Scikit-Decide, the Open-Source C++ and Python Library for
Planning, Scheduling and Reinforcement Learning"
layout: page
hero_height: is-fullwidth
---

### Summary

[Scikit-decide](https://github.com/airbus/scikit-decide)
is an open-source library for modeling and solving planning, scheduling and
reinforcement learning problems within a common API which helps break technical
silos between different decision-making communities and enables seamless
benchmarking of different approaches. For instance, one can solve PDDL problems
with both classical planning (via a bridge to
[Unified Planning](https://github.com/aiplan4eu/unified-planning)) and
reinforcement learning (via a bridge to
[RLlib](https://docs.ray.io/en/latest/rllib/index.html)) solvers with very
few lines of code, and compare the different solutions. Thinking of both
algorithm providers and solver users, the library's class hierarchy has been
designed to ease the integration of new domains and algorithms depending on
their distinctive features (e.g. partially vs fully observable states,
deterministic vs probabilistic state transitions, single vs multi agents,
simulation-based vs formal transition models, etc.).

With more than 125k total
downloads and 200 downloads per day on
[PyPi](https://pypi.org/project/scikit-decide/), the library is
gaining traction in the global sequential decision-making landscape, including
practitioners and researchers. It is officially sponsored by
[ANITI](https://aniti.univ-toulouse.fr/en/) (the Artifical and Natural
Intelligence Toulouse Institute) and is the main host for the research
algorithms produced in the Horizon Europe's [TUPLES](https://tuples.ai/)
project (Trustworthy Planning and Scheduling with Learning and Explanations).

<span style="color:darkblue"><b>The half-day tutorial will show how to model and solve the same problems using
algorithms from different communities, and how to extend the libraries with new
domains and solvers in a few lines of code. It will alternate presentations and
live Python coding sessions.</b></span>

### Agenda

- **Introduction** (15 mn): General concepts of the library: domains, solvers, spaces, hub, features
- **Part I** (90mn) : Solving domains (aka problems) with auto-selected compatible solvers
   - Notebook I : Solving control problems with reinforcement learning, and width-based planning solvers
   - Notebook II : Solving scheduling problems with constraint programming, operation research, and reinforcement learning solvers
   - Notebook III : Solving PDDL problems with classical planning, and reinforcement learning solvers
- **Part II** (60mn) : Implementing your own domains and solvers
   - Notebook IV : Implementing a scikit-decide domain for RDDL problems
   - Notebook V : Implementing a scikit-decide solver embedding the PROST planner and solving RDDL-based scikit-decide domains
- **Conclusion** (15mn) : Applications, contribution guidelines, and future developments