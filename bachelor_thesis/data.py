"""
Module containing the ``pytorch_lightning.LightningDataModule`` implementation
of the package.
"""
import os.path
import pickle
import string

import gensim
import h5py
import progressbar
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchtext

from .dataset import BachelorThesisDataset


class BachelorThesisData(pl.LightningDataModule):
    """Implementation of the ``pytorch_lightning.LightningDataModule`` used in
    this package.

    Attributes:
        data: Object containing the data in H5 format.
        words_list: List of words sorted by frequency.
        words_embeddings: List of word embeddings in the same order as
        ``self.words_list``.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.data = h5py.File(kwargs['data_path'], 'r')
        self.words_list, self.words_embeddings = None, None
        self.kwargs = kwargs

    def prepare_data(self):
        """Prepares the data used to run the package.

        Fills ``self.words_dict`` and ``self.words_embeddings``, containing
        the ``--dict_size`` most common words in the dataset. The result is
        stored in ``--data_ckpt_path`` the first time the method is called and
        restored every time its called again.
        """
        if os.path.exists(self.kwargs['data_ckpt_path']):
            with open(self.kwargs['data_ckpt_path'], 'rb') as data_ckpt:
                self.words_list, self.words_embeddings = pickle.load(data_ckpt)
        else:
            words_dict = self._get_words_frequency()
            self.words_list, self.words_embeddings = \
                self._get_words_list_embeddings(words_dict)
            with open(self.kwargs['data_ckpt_path'], 'wb') as data_ckpt:
                pickle.dump(
                    (self.words_list, self.words_embeddings), data_ckpt
                )

    def _get_words_frequency(self):
        """Returns a dictionary mapping the words in the entire dataset
        along with the number of times they appear.

        Returns:
            Dictionary mapping words and their frequency.
        """
        words_dict = {}
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

        available_dates = self.data['news/reuters'].keys()
        for i, day in progressbar.progressbar(enumerate(available_dates)):
            day_news = self.data['news/reuters'][day][()]
            for new_index in range(day_news.shape[0]):
                for new_word in tokenizer(
                        day_news[new_index, 0].decode('utf-8').translate(
                            str.maketrans('', '', string.punctuation)
                        )
                ):
                    words_dict[new_word] = words_dict[new_word] + 1 \
                        if new_word in words_dict else 1
        return words_dict

    def _get_words_list_embeddings(self, words_dict):
        """Returns the most common ``--dict_size`` words in the dataset
        along with its word embeddings using Word2Vec.

        Args:
            words_dict: Dictionary associating words with the number of times
            they appear in the data.

        Returns:
            Tuple of two positions containing:
                - A list of the first ``--dict_size`` words.
                - The word embeddings associated with those words, in the same
                    order.
        """
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            self.kwargs['word2vec_path'], binary=True
        )

        words_list = ['<EMP>', '<UNK>']
        words_embeddings = [torch.zeros((300,)), torch.ones((300,))]
        for word in sorted(
                words_dict.keys(), key=lambda x: words_dict[x], reverse=True
        ):
            if word in word2vec:
                words_list.append(word)
                words_embeddings.append(
                    torch.from_numpy(word2vec[word])
                )
            if len(words_list) >= self.kwargs['dict_size']:
                break
        return words_list, torch.stack(words_embeddings)

    def train_dataloader(self):
        """Returns the data loader containing the training data.

        Returns:
            Data loader containing the training data.
        """
        train_dataset = BachelorThesisDataset(
            self.data, self.words_list, 'train', **self.kwargs
        )
        return torch.utils.data.DataLoader(
            train_dataset, self.kwargs['batch_size'], True,
            num_workers=self.kwargs['num_workers']
        )

    def val_dataloader(self):
        """Returns the data loader containing the validation data.

        Returns:
            Data loader containing the validation data.
        """
        validation_dataset = BachelorThesisDataset(
            self.data, self.words_list, 'validation', **self.kwargs
        )
        return torch.utils.data.DataLoader(
            validation_dataset, self.kwargs['batch_size'],
            num_workers=self.kwargs['num_workers']
        )

    def test_dataloader(self):
        """Returns the data loader containing the test data.

        Returns:
            Data loader containing the test data.
        """
        test_dataset = BachelorThesisDataset(
            self.data, self.words_list, 'test', **self.kwargs
        )
        return torch.utils.data.DataLoader(
            test_dataset, self.kwargs['batch_size'],
            num_workers=self.kwargs['num_workers']
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """Adds data-related CLI arguments to the parser.

        Args:
            parent_parser: Parser object just before adding the arguments.

        Returns:
            Parser object after adding the arguments.
        """
        parser = parent_parser.add_argument_group('BachelorThesisData')
        parser.add_argument(
            '--data_path', default='./data/bachelor_thesis_data.hdf5'
        )
        parser.add_argument(
            '--data_ckpt_path', default='./lightning_logs/data.ckpt'
        )
        parser.add_argument('--word2vec_path', default='./data/word2vec.bin')
        parser.add_argument('--dict_size', type=int, default=10000)
        parser.add_argument('--n_news', type=int, default=25)
        parser.add_argument('--max_words', type=int, default=15)
        parser.add_argument('--symbol', default='AAPL')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--train_years', type=int, nargs='+',
                            default=[2010, 2011, 2012, 2013, 2014, 2015, 2016])
        parser.add_argument('--validation_years', type=int, nargs='+',
                            default=[2017])
        parser.add_argument('--test_years', type=int, nargs='+',
                            default=[2018])
        return parent_parser
