# LOGML project on Graph Learning for Uplift Modeling

Start by installing the requirements.

```bash
pip install -r requirements.txt
```


## Data
The script prepare_data downloads and processes the [RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) into the folder data/retailhero and the [Movielens25](https://grouplens.org/datasets/movielens/25m/) into folder data/movielens.

```bash
python prepare_data.py
```

## Methods

You can start from the 'logml_start.ipynb' to see a step by step breakdown of the project and the initial results for 1 of the 10-folds evaluation. Then you can run 

```bash
python logml_start.py
```

which uses the functions introduced in the notebook and runs the experiment to get an initial comparison between the GNN model and a few baseline methods.
