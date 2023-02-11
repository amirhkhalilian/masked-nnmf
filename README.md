# masked non-negative matrix factorization

Non-negative matrix factorization is a powerful tool for dimensionality reduction and data analysis. Here I have implemented a solver that takes missing entries into account when solving the NNMF problem. The solver is implemented in Matlab and use the method proposed in [this paper](https://ieeexplore.ieee.org/document/4781130) to solve non-negative least squares problem. The code for the paper can be fount [here](http://www.cc.gatech.edu/~hpark/software/nmf_bpas.zip).  We use `nnlsm_blockpivot.m` and `solveNormalEqComb.m` files from this package. These files are added under solvers.

This solver closely follows the methods introduced in [this blog post](http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/) and [this paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3312-5).

## masked non-negative matrix factorization problem

Given the non-negative data matrix $D\in\mathbb{R}^{m\times n}_{\geq 0}$ we want to find matrices $W\in\mathbb{R}^{m\times r}_{\geq 0}$ and $H\in\mathbb{R}^{r\times n}_{\geq 0}$  such that $D\approx WH$. The optimization problem can be written as 

$$ W^{\ast}, H^{\ast} = \arg\min_{W,H} \|D-WH\|_{F}^2 \quad s.t. \quad W\in\mathbb{R}^{m\times r},\quad W\geq 0 \quad and \quad H \in \mathbb{R}^{r\times n},\quad H\geq 0.$$


