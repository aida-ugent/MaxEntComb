# Block-Approximated Exponential Random Graphs

This repository contains the source code, installation and use instructions for the methods presented in the paper: 
*Block-Approximated Exponential Random Graphs*. Instructions for replicating 
the experiments in the paper are also given.

We provide two different implementations, a Python version for the MaxEnt model with full structural constraints and
a Matlab version for the MaxEnt using low rank approximations.

## Installation

The Python portion of this repository can be installed directly from GitHub with:

```shell
$ pip install git+https://github.com/aida-ugent/MaxEntComb.git
```

It can be installed from the source with:

```shell
$ git clone https://github.com/aida-ugent/MaxEntComb.git
$ cd MaxEntComb
$ pip install -e .
```

Where `-e` means "editable" mode.

The MaxEnt Matlab version using low rank approximations requires `Matlab R2018b` or higher. The code also uses the open
source [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) package for unconstrained differentiable 
multivariate optimization by M. Schmidt. For convenience, this packages is already included in the `/matlab` folder.

## Usage

### Running MaxEnt Python:
A `maxentcomb` CLI command is added on `pip install` which runs the model. The file takes the following
parameters:
```text
  -h, --help                Show this help message and exit.
  --inputgraph [INPUTGRAPH] Input graph path.
  --output [OUTPUT]         Path where the embeddings will be stored. Default is `network.emb`.
  --tr_e [TR_E]             Path of the input train edges. Default is None (in this case returns embeddings)
  --tr_pred [TR_PRED]       Path where the train predictions will be stored. Default is `tr_pred.csv`
  --te_e [TE_E]             Path of the input test edges. Default is None.
  --te_pred [TE_PRED]       Path where the test predictions will be stored. Default is `te_pred.csv`
  --learning_rate LEARNING_RATE
                            Learning rate. Default is 0.1.
  --epochs EPOCHS           Training epochs. Default is 100.
  --grad_min_norm GRAD_MIN_NORM
                            Early stop if grad norm is below this value. Default is 0.0001.
  --optimizer OPTIMIZER     Optimizer to use. Options are `newton` and `grad_desc`. Default is `newton`.
  --memory MEMORY           The constraints on the memory to use. Options are `quadratic` and `linear`. 
                            Quadratic is considerably faster but used O(n^2) memory. Default is `quadratic`.
  --prior PRIOR [PRIOR ...] The prior to use. Options are `CN`, `AA`, `A3`, `JC`, `PA` or list combining them. 
                            Default is [`CN`].
  --delimiter DELIMITER     The delimiter used in the input edgelist. Output will use the same one. Default is `,`
  --dimension DIMENSION     Not used. Default is 1.
  --verbose                 Determines the verbosity level of the output. Default is False.
```

**NOTE:** The user can define new constraint matrices *F*. However, these need to implement the interfaces provided 
in the `weighted_lin_constr.py` file.

Examples of running the Python MaxEnt model are:
```bash
# Example 1: Outputs posterior matrix after calculation
maxentcomb --inputgraph ./graph.edgelist --prior 'CN' 'PA' --optimizer 'newton'
# Example 2: Outputs predictions for the edge pairs in the tr_e and te_e files
maxentcomb --inputgraph ./graph.edgelist --tr_e ./tr.edgelist --te_e ./te.edgelist --tr_pred './tr.out' --te_pred './te.out'
```

### Running MaxEnt Matlab:

The file `test.m` provides an overview on how to run a complete pipeline using the Matlab MaxEnt code.
This code generates a symmetric adjacency matrix and computes its spectral decomposition. Second and third order
proximity matrices are then computed, as well as low rank approximations for Resource Allocation Index and Preferential 
Attachment. A cartesian product of these block approximation is then obtained and the MaxEnt model is fitted on this 
binning using the *lbfgs* solver from the minFunc package. Finally train and test predictions are computed for the 
input edge pairs. 

In order to simplify the execution of the Matlab MaxEnt version, we also provide a pre-compiled file named `test` as 
well as a shell script `run_test.sh` for running it. The shell script takes the following parameters in this order:
*inputgraph, tr_e, te_e, tr_pred, te_pred, dim, bins, weights*. The descriptions of the first 5 parameters are the 
same as for the Python version. As for the remaining ones:

```text
dim:        FLOAT       Embedding dimensionality
bins:       FLOAT       Number of bins each F matrix is divided in.
weights:    LIST        The multiplicity for each high-order priximity of A. 
                        E.g. weights[1]*A^2 + weights[2]*A^3 ...
```
 
An example of running the Matlab MaxEnt version:
```bash
./run_test.sh /usr/local/MATLAB/MATLAB_Runtime/v95 graph.edgelist ./tr.edgelist ./te.edgelist './tr.out' './te.out' 8 100 [1,0.1]
``` 

## Reproducing Experiments
In order to reproduce the experiments presented in the paper the following steps are necessary:

1. Download and install the EvalNE library as instructed [here](https://github.com/Dru-Mara/EvalNE)
2. Download and install the MaxEnt method as shown in the [Installation](#Installation) section above
3. Download and install the baseline methods. We recommend that each method is installed in 
a unique virtual environment to ensure that the right dependencies are used. For all methods except SDNE, we use the
implementations provided by the original authors: 
 
    * [Deepwalk](https://github.com/phanein/deepwalk),
    * [Node2vec](https://github.com/aditya-grover/node2vec),
    * [LINE](https://github.com/tangjianpku/LINE),
    * [Struc2vec](https://github.com/leoribeiro/struc2vec),
    * [AROPE](https://github.com/ZW-ZHANG/AROPE),
    * [CNE](https://bitbucket.org/ghentdatascience/cne/).

    For SDNE the implementation available [here](https://github.com/palash1992/GEM) 
is the only one which has shown similar results to those in the original paper.

4. Download the datasets: 

    * [StudentDB](http://adrem.ua.ac.be/smurfig), 
    * [Facebook](https://snap.stanford.edu/data/egonets-Facebook.html), 
    * [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3), 
    * [Flickr](http://socialcomputing.asu.edu/datasets/Flickr),
    * [YouTube](http://socialcomputing.asu.edu/datasets/YouTube2),
    * [GR-QC](https://snap.stanford.edu/data/ca-GrQc.html),
    * [DBLP](https://snap.stanford.edu/data/com-DBLP.html),
    * [PPI](http://snap.stanford.edu/node2vec/#datasets) and
    * [Wikipedia](http://snap.stanford.edu/node2vec/#datasets).

5. Download the `.ini` configuration files from the `/experiments` folder. Modify the *dataset* paths and *method* paths to
match the respective installation directories and run each files as:

    ```bash
    python -m evalne ./experiments/conf_maxent.ini
    ```

**NOTE:** For AROPE a `main.py` file is required in order to run the evaluation through EvalNE. This file
is also available in the `/experiments` folder.


## Citation ##

If you have found our research useful, please consider citing our [paper](https://ieeexplore.ieee.org/document/9260034), 
which is also available on [arxiv](https://arxiv.org/abs/2002.07076):

```bibtex
@INPROCEEDINGS{9260034,
  author={F. {Adriaens} and A. {Mara} and J. {Lijffijt} and T. {De Bie}},
  booktitle={2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={Block-Approximated Exponential Random Graphs}, 
  year={2020},
  pages={70-80},
  doi={10.1109/DSAA49011.2020.00019}}
```
