ProfLogit for Customer Churn Prediction
=======================================

*A profit maximizing logit model for churn prediction.*


Purpose
-------

ProfLogit builds a logistic regression model that maximizes the [expected
maximum profit measure for customer churn (EMPC)](http://ieeexplore.ieee.org/document/6165289/)
through a real-coded genetic algorithm (RGA).


Abstract
--------

The accompanied paper entitled "*Profit Maximizing Logistic Model for Customer
Churn Prediction Using Genetic Algorithms*"
[is published](http://authors.elsevier.com/sd/article/S2210650216301754) in the international peer-reviewed journal of
[Swarm and Evolutionary Computation](https://www.journals.elsevier.com/swarm-and-evolutionary-computation).


> To detect churners in a vast customer base, as is the case with telephone
service providers, companies heavily rely on predictive churn models to remain
competitive in a saturated market. In previous work, the expected maximum profit
measure for customer churn (EMPC) has been proposed in order to determine the
most profitable churn model. However, profit concerns are not directly integrated
into the model construction. Therefore, we present a classifier, named ProfLogit,
that maximizes the EMPC in the training step using a genetic algorithm, where
ProfLogit's interior model structure resembles a lasso-regularized logistic model.
Additionally, we introduce threshold-independent recall and precision measures
based on the expected profit maximizing fraction, which is derived from the EMPC
framework. Our proposed technique aims to construct profitable churn models for
retention campaigns to satisfy the business requirement of profit maximization.
In a benchmark study with nine real-life data sets, ProfLogit exhibits the overall
highest, out-of-sample EMPC performance as well as the overall best, profit-based
precision and recall values. As a result of the lasso resemblance, ProfLogit also
performs a profit-based feature selection in which features are selected that would
otherwise be excluded with an accuracy-based measure, which is another noteworthy
finding.


Citation
--------

If you find ProfLogit useful, please cite it in your publications.
You can use the following [BibTeX](http://www.bibtex.org/) entry:

```
@article{stripling2018proflogit,
  title={{Profit Maximizing Logistic Model for Customer Churn Prediction Using Genetic Algorithms}},
  author={Stripling, Eugen and vanden Broucke, Seppe and Antonio, Katrien and Baesens, Bart and Snoeck, Monique},
  journal={Swarm and Evolutionary Computation},
  volume={40},
  pages={116-130},
  year={2018},
  issn={2210-6502},
  publisher={Elsevier},
  keywords={Data mining, Customer churn prediction, Lasso-regularized logistic regression model, Profit-based model evaluation, Real-coded genetic algorithm},
  doi={10.1016/j.swevo.2017.10.010},
}
```


License
-------
The code in this repository, including all code samples in the accompanied
notebooks, is released under the GNU General Public License v3 (GPLv3).


Installation
------------

```{bash}
git clone https://github.com/estripling/proflogit.git
cd proflogit/
python3 -m pip install -r requirements.txt --no-cache-dir
python3 -m pip install .
```


Examples
--------
Examples are given in [notebooks](https://github.com/estripling/proflogit/tree/master/notebooks) directory.
