# lovebirds

# ![lovebirds-title](lovebirds-title.svg)

A Python package for simulating ecological communities and their biased or unbiased observations, across space and time

_lovebirds_ 'reverse-engineers' generalised dissimilarity modelling (GDM)
to simulate communities distributed across heterogeneous landscapes,
then uses a 'virtual ecologist' approach to simulate various observation processes on those communities.

![overview](img/sim_comm_comp_design.png)

## Overview

This package is designed for methods development in community ecology and biodiversity
modelling. It 'reverse-engineers' generalised dissimilarity modelling (GDM), allowing
the user to define environmental landscapes, monotonic environmental
turnover functions (e.g. I-splines) that relate ecological community turnover to those landscapes' variables, the size of the regional species pool (i.e., gamma diversity), and the set of sampling locations. Then it simulates communities at all sampling locations and 
uses a 'virtual ecologist' approach to simulate various observation processes on those communities
(ranging from full-communtiy censuses, to abudance-absence, presence-absence, or presence-only (i.e., 'opportunistic') records.

Workflows can be built from the following steps:

1. Define a landscape and environmental turnover functions.
2. Simulate the latent communities.
3. Simulate one or more observation/survey processes.
4. Fit/visualize GDM or other biodiversity models to the simulated data.
5. Modify environmental layers to represent environmental change.
6. Re-simulate communities and observations.

See the documentation for the full API and examples.

## Installation

```bash
pip install lovebirds
```

## License

See `LICENSE`.
