.. local2global documentation master file, created by
   sphinx-quickstart on Wed Aug 11 10:54:55 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
========================================
.. toctree::
   :hidden:

   self


Local2Global
_____________

Python implementation of the ``local2global`` patch embedding synchronisation
method used in [#l2g]_. This code uses embeddings for a set of overlapping
patches as input and aligns them to obtain a global embedding. The alignment step is based on the eigenvector synchronisation method of [#eigsync]_. Source code is hosted on `GitHub <https://github.com/LJeub/Local2Global.git>`_.


References
+++++++++++

.. [#l2g] \L. G. S. Jeub, G. Colavizza, X. Dong, M. Bazzi, M. Cucuringu (2021). Local2Global: Scaling global representation learning on graphs via local training. DLG-KDD'21. `arXiv:2107.12224 [cs.LG] <https://arxiv.org/abs/2107.12224>`_

.. [#eigsync] \M. Cucuringu, Y. Lipman, A. Singer (2012). Sensor network localization by eigenvector synchronization over the euclidean group. ACM Transactions on Sensor Networks 8.3. DOI: `10.1145/2240092.2240093 <https://doi.org/10.1145/2240092.2240093>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage/usage
   test
   reference




Index
_______________________

* :ref:`genindex`
