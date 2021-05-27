# Bachelor Thesis
[![](https://img.shields.io/github/v/release/davidalvarezdlt/bachelor_thesis)](https://github.com/davidalvarezdlt/bachelor_thesis/releases)
[![](https://img.shields.io/badge/python-%3E3.7-blue)](https://www.python.org/)
[![](https://requires.io/github/davidalvarezdlt/bachelor_thesis/requirements.svg)](https://requires.io/github/davidalvarezdlt/bachelor_thesis/requirements/)
[![](https://www.codefactor.io/repository/github/davidalvarezdlt/bachelor_thesis/badge)](https://www.codefactor.io/repository/github/davidalvarezdlt/bachelor_thesis)
[![](https://img.shields.io/github/license/davidalvarezdlt/bachelor_thesis)](https://github.com/davidalvarezdlt/bachelor_thesis/blob/main/LICENSE)

This repository contains a partial implementation of my bachelor's thesis ["Real-time stock predictions with Deep Learning and news scrapping"](https://upcommons.upc.edu/handle/2117/128164).
While the data and the cleaning pipeline is exactly as described in the report,
the models and the results are not shared.

## About the Data

The data used in the thesis has been completely crawled and put together from
scratch. Specifically, you can find the titles and descriptions of the news
published in [reuters.com](https://www.reuters.com) from January 2010 to May
2018. In addition to that, you also have the stock prices at the end of the day
of S&P 500 companies extracted from [alphavantage.co](https://www.alphavantage.co).

Everything is compressed in a H5DF file that you can download from
[this link](https://www.kaggle.com/davidalvarezdlt/bachelor-thesis) (3.93 GB).

In order to access the data, you must load it using ``h5py``. You can then get
the news of a certain date or the stock price movements of one of the symbols
as:

```
data = h5py.File('path/to/bachelor_thesis_data.hdf5', 'r')
date_news = data['news/reuters']['2010-01-20'][()]
stock_prices = data['prices/AAPL']['2010-01-20'][()]
```

For the case of the news, ``date_news`` is a ``np.ndarray`` of size
``(n_news, 5)`` containing the title, description, category, URL and UTC
publishing datetime of the news of that specific date.

For the case of the stock prices, ``stock_prices`` is also a ``np.ndarray`` of
size ``(8,)`` containing the opening price, maximum price, minimun price,
closing price, volume of traded stocks, dividend and split coefficient.

Notice that not every date is available, both in the case of the news and the
stock prices. Read the documentation of HDF5 to learn more about how to deal
with this type of files.

## Running an Experiment

The first step is to clone this repository in your computer and install its
dependencies:

```
git clone https://github.com/davidalvarezdlt/bachelor_thesis.git
cd bachelor_thesis
pip install -r requirements.txt
```

After downloading the repository to your personal computer, make sure to move
the data to ``bachelor_thesis/data/``. You will also have to download [Word2Vec](https://www.kaggle.com/davidalvarezdlt/bachelor-thesis)
and store it in the same path, as it's used to get the word embeddings. The
file structure must be:

```
bachelor_thesis/
    bachelor_thesis/
    data/
        bachelor_thesis_data.hdf5
        word2vec.bin
    experiments/
    config.default.json
    README.md
    requirements.txt
```

The implementation of this repository is done using [Skeltorch](https://github.com/davidalvarezdlt/skeltorch).
Read its documentation to get a complete overview of how is this repository
organized. The first step would be to create a new experiment:

```
python -m bachelor_thesis --experiment-name test --verbose init --config-path config.default.json
```

This will create the folder ``bachelor_thesis/experiments/test/`` with the
files required to run the experiment. Notice that the configuration
file ``config.default.json`` contains several important parameters related to
both the data and the model. You can then train the toy model calling:

```
python -m bachelor_thesis --experiment-name test --verbose train --device cuda
```

The experiment will start training using a GPU. If you don't have one, make
sure to change to ``--device cpu``. Notice that the only goal of the model of
this repository is to provide a closed pipeline and it will definitely not
converge.

By default, each data item contains the news of a certain date. This would not
be a realistic training methodology for a live trading agent, as it contains
data which is published after the trading session. Make sure to modify this if
that is your intention. If you manage to make the model converge, remember to
send me an email so we can share the benefits.

## Citation

If you use the data provided in this repository or if you find this thesis
useful, please cite the following report:

```
@thesis{Alvarez2018,
    type = {Bachelor's Thesis},
    author = {David Álvarez de la Torre},
    title = {Real-time stock predictions with Deep Learning and news scrapping},
    school = {Universitat Politècnica de Catalunya},
    year = 2018,
}
```
