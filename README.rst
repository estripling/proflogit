ProfLogit with RGA optimizer
============================

*A profit maximizing classifier for predictive customer churn modeling.*


Purpose
-------

ProfLogit builds a logistic regression model that maximizes the expected
maximum profit measure for customer churn (EMPC) through a real-coded genetic
algorithm (RGA).


Usage
-----

To create a stand-alone Python 3.6 environment named `proflogit-rga`,
run the following::

    $ conda create -n proflogit-rga python=3.6 --file requirements.txt

To activate the `proflogit-rga` environment, run the following (on Linux)::

    $ source activate proflogit-rga

To install the library (*and all development dependencies*), run the following::

    $ pip install -e .[test]

To run all tests, execute the following command::

    $ python setup.py test

This will trigger `py.test <http://pytest.org/latest/>`_, along with its popular
`coverage <https://pypi.python.org/pypi/pytest-cov>`_ plugin.


Note
----

The accompanied paper entitled *Profit Maximizing Logistic Model for Customer Churn Prediction Using Genetic Algorithms*
has been accepted by the international peer-reviewed journal of
`Swarm and Evolutionary Computation <https://www.journals.elsevier.com/swarm-and-evolutionary-computation>`_.