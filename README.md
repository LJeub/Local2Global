# Local2Global

Python implementation of the ``local2global`` patch embedding synchronisation
method used in [[1]](#1). This code uses embeddings for a set of overlapping
patches as input and aligns them to obtain a global embedding. The alignment step is based on the eigenvector synchronisation method of [[2]](#2).


## Installation

The `local2global` package can be installed using `pip`. Simply run

```
pip install git+ssh://git@github.com/LJeub/Local2Global.git@master  
```


## Usage

For more information see the [documentation](https://ljeub.github.io/Local2Global/).

## References

<a id="1">[1]</a> L. G. S. Jeub, G. Colavizza, X. Dong, M. Bazzi, M. Cucuringu (2021). Local2Global: Scaling global representation learning on graphs via local training. DLG-KDD'21. [arXiv:2107.12224 [cs.LG]](https://arxiv.org/abs/2107.12224)

<a id="2">[2]</a> M. Cucuringu, Y. Lipman, A. Singer (2012). Sensor network localization by eigenvector synchronization over the euclidean group. ACM Transactions on Sensor Networks 8.3. DOI: [10.1145/2240092.2240093](https://doi.org/10.1145/2240092.2240093)

