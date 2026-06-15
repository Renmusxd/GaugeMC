# GaugeTheoryMC

A library for 4D GPU simulations of a compact U (1) gauge theory in the dual integer representation

In the dual representation, the partition function is given by
$$Z = \sum_{\{n_p\}} e^{-\sum_p \tilde{V}_{n_p}} \left (\prod_e \delta\left[\sum_p d_{ep} n_p\right]\right)$$
where each plaquette $p$ hosts an integer $n_p$ with action cost $V\tilde{V}_{n_p}$. At each edge $e$ there is a
continuity condition $d_{ep} n_p = 0$, with $d_{pe}$ the oriented adjacency matrix.

## Installation and Testing

The bulk of the library is implemented in **cuda**, which must be independently installed on your system. Other
dependencies are installed automatically with **cargo**

To check your installation run

```bash
cargo test
```

It's not uncommon for a small number (1 or 2) of tests to fail if your system is hosting a GPU with few registers. This
may indicate you need to use a smaller thread block size (env variable `GAUGEMC_BLOCK_SIZE` defaults to 1024).

