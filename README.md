# Real-time stock predictions with deep learning and news scraping
[![](https://img.shields.io/badge/publication-UPC%20Commons-red)](https://upcommons.upc.edu/handle/2117/128164)
[![](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![](https://www.codefactor.io/repository/github/davidalvarezdlt/bachelor_thesis/badge)](https://www.codefactor.io/repository/github/davidalvarezdlt/bachelor_thesis)
[![](https://img.shields.io/github/license/davidalvarezdlt/bachelor_thesis)](https://github.com/davidalvarezdlt/bachelor_thesis/blob/main/LICENSE)

This repository contains a partial implementation of my bachelor's thesis ["Real-time stock predictions with deep learning and news scraping"](https://upcommons.upc.edu/handle/2117/128164).
The code has been built using PyTorch Lightning, read its documentation to get a complete overview of how this repository is structured.

**Disclaimer**: Neither the pipeline nor the model published in this repository
are the ones used in the thesis. On the pipeline side, notice that the model
tries to match headlines and prices of the same day, while in the thesis we
used news published the day before. For the case of the model, the one shared
here has nothing to do with the original and should be considered a toy model.

## Preparing the data

The data used in the thesis has been completely crawled and put together from
scratch. Specifically, you can find the titles and descriptions of the news
published on Reuters.com from January 2010 to May 2018. In addition to that,
you also have the stock prices (end of the day) of S&P 500 companies extracted
from AlphaVantage.co. Everything is compressed in a H5DF file that you can
download from [this link](https://www.kaggle.com/davidalvarezdlt/bachelor-thesis).

The first step is to clone this repository and install its dependencies:

```
git clone https://github.com/davidalvarezdlt/bachelor_thesis.git
cd bachelor_thesis
pip install -r requirements.txt
```

Move both ``bachelor_thesis_data.hdf5`` and ``word2vec.bin`` inside ``./data``.
The resulting folder structure should look like this:

```
bachelor_thesis/
    bachelor_thesis/
    data/
        bachelor_thesis_data.hdf5
        word2vec.bin
    lightning_logs/
    .gitignore
    .pre-commit-config.yaml
    LICENSE
    README.md
    requirements.txt
```

## Training the model

In short, you can train the model by calling:

```
python -m bachelor_thesis
```

You can modify the default parameters of the code by using CLI parameters. Get
a complete list of the available parameters by calling:

```
python -m bachelor_thesis --help
```

For instance, if we want to train the model using ``GOOGL`` stock prices,
with a batch size of 32 and using one GPUs, we would call:

```
python -m bachelor_thesis --symbol GOOGL --batch_size 32 --gpus 1
```

Every time you train the model, a new folder inside ``./lightning_logs`` will
be created. Each folder represents a different version of the model, containing
its checkpoints and auxiliary files.

## Testing the model

You can measure the loss and the accuracy of the model (number of times the
prediction is correct) and store it in TensorBoard by calling:

```
python -m bachelor_thesis --test --test_checkpoint <test_checkpoint>
```

Where ``--test_checkpoint`` is a valid path to the model checkpoint that should
be used.

## Citation

If you use the data provided in this repository or if you find this thesis
useful, please use the following citation:

```
@thesis{Alvarez2018,
    type = {Bachelor's Thesis},
    author = {David Álvarez de la Torre},
    title = {Real-time stock predictions with Deep Learning and news scrapping},
    school = {Universitat Politècnica de Catalunya},
    year = 2018,
}
```
