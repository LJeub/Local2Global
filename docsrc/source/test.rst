Tests
=====

The ``local2global`` package includes tests that check reconstruction performance on synthetic data.
The test use `pytest <https://pytest.org>`_ and are contained in :py:mod:`local2global.test_local2global`.

To run the tests use

.. code-block:: bash

    python -m local2global.test_local2global [args]

where any command-line arguments are passed along to ``pytest``.
For details of the test see :py:mod:`~local2global.test_local2global`.
