"""
Module containing the ``torch.utils.data.Dataset`` implementation of the
package.
"""
import random
import re
import string

import torch.nn.utils.rnn
import torch.utils.data
import torchtext


class BachelorThesisDataset(torch.utils.data.Dataset):
    """Implementation of the ``torch.utils.data.Dataset`` used in this package.

    Attributes:
        words_list: List of words sorted by frequency.
        split: Identifier of the data split. Can be either "train",
            "validation" or "test".
        items_indexes: List of available dates in YYYY-MM-DD format.
        news: List of the same size of ``self.items_indexes`` containing the
            news published at the dates given in such days.
        prices: List of the same size of ``self.items_indexes`` containing the
            different prices of the stock given in ``--symbol``.
        tokenizer: Tokenizer object used to tokenize sentences.
        kwargs: Dictionary containing the CLI arguments used in the execution.
    """
    def __init__(self, data, words_list, split, **kwargs):
        self.words_list = words_list
        self.split = split
        self.items_indexes, self.news, self.prices = [], [], []
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        self.kwargs = kwargs
        self._init_items_indexes(data)

    def _init_items_indexes(self, data):
        """Fills the paramters ``self.items_indexes``, ``self.news`` and
        ``self.prices``.

        Args:
            data: Object containing the data in H5 format.
        """
        if self.split == 'train':
            years = self.kwargs['train_years']
        elif self.split == 'validation':
            years = self.kwargs['validation_years']
        else:
            years = self.kwargs['test_years']

        news_items_list = list(data['news/reuters'].keys())
        items_cands = filter(
            lambda x: int(x[:4]) in years,
            data['prices/{}'.format(self.kwargs['symbol'])].keys()
        )

        for item in items_cands:
            if item in news_items_list and len(
                    data['news/reuters'][item][()]
            ) >= self.kwargs['n_news']:
                self.items_indexes.append(item)

        for item_index in self.items_indexes:
            self.news.append(data['news/reuters'][item_index][()])
            self.prices.append(data['prices/{}'.format(
                self.kwargs['symbol']
            )][item_index][()])

    def __getitem__(self, item):
        """Returns the news and the movement to predict of a certain date.

        Returns a set of ``--n_news`` news along with a binary value
        representing if the stock has increased or decreased its price during
        that date.

        Args:
            item: Index of the date in ``self.items_indexes``.

        Returns:
            Tuple of three positions containing:
                - A tensor of size (``len(news_titles)``,``--max_words``)
                    contaning the one-hot embedding of the first
                    ``--max_words`` of the headlines given in ``news_titles``.
                - A tensor of size ``len(news_titles)`` containing the real
                    length of the headlines given in ``news_titles``.
                - A 1 if the stock price has increased its price,
                    0 if it has decreased.
        """
        item_news, item_price = self.news[item], self.prices[item]
        news_titles = [
            item_news[new_index][0]
            for new_index in random.sample(
                list(range(item_news.shape[0])), self.kwargs['n_news']
            )
        ]

        x, x_length = self._pack_news_titles(news_titles)
        y = torch.tensor(1) \
            if item_price[3] > item_price[0] else torch.tensor(0)

        return x, x_length, y

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            Length of the dataset, given by the number of available dates in
                ``self.items_indexes``.
        """
        return len(self.items_indexes)

    def _pack_news_titles(self, news_titles):
        """Packs raw headlines into one-hot embeddings after applying a
        cleaning pipeline.

        Args:
            news_titles: List of raw headlines.

        Returns:
            Tuple of two positions containing:
                - A tensor of size ``(len(news_titles), --max_words)``
                    contaning the one-hot embedding of the first
                    ``--max_words`` of the headlines given in ``news_titles``.
                - A tensor of size ``len(news_titles)`` containing the real
                    length of the headlines given in ``news_titles``.
        """
        news_indexes = torch.zeros(
            (1, self.kwargs['n_news'], self.kwargs['max_words']),
            dtype=torch.int64
        )
        news_lengths = torch.zeros(
            (1, self.kwargs['n_news']), dtype=torch.int64
        )

        for i, new_title in enumerate(news_titles):
            new_title_tokenized = self._clean_and_tokenize(
                new_title.decode('utf-8')
            )[:self.kwargs['max_words']]
            for j, new_word in enumerate(new_title_tokenized):
                news_indexes[0, i, j] = self.words_list.index(new_word) \
                    if new_word in self.words_list else 1
            news_lengths[0, i] = len(new_title_tokenized)

        return news_indexes.squeeze(0), news_lengths.squeeze(0)

    def _clean_and_tokenize(self, text):
        """Cleans the input string ``text``.

        Cleans a Reuters-specific new headline following the procedure
        explained in the report of the thesis. In short:

            1. Removes the introductory text of many Reuter headlines,
                if present.
            2. Removes possible non-informative tags at the end of the
                headline.
            3. Transforms the text to lowercase.
            4. Removes points that may be present.
            5. Removes numbers that may be present.
            6. Removes commas that may be present.
            7. Replaces special characters with spaces.
            8. Removes non-single spaces.
            9. Removes possible spaces at the beginning and the end of the
                headline (inserted by previous steps).

        Args:
            text: Reuters new headline before being cleaned.

        Returns:
            Reuters new headline after being cleaned.
        """
        text_sanitized = re.sub(r'^[A-Z0-9-\s]*-', '', text)
        text_sanitized = re.sub(r'\s-[\s\w]*$', '', text_sanitized)
        text_sanitized = text_sanitized.lower()
        text_sanitized = re.sub(r'[\.]+', '', text_sanitized)
        text_sanitized = re.sub(r'\w*\d\w*', '', text_sanitized)
        text_sanitized = re.sub(r'[\']', '', text_sanitized)
        text_sanitized = re.sub(r'[^a-z0-9\s]', ' ', text_sanitized)
        text_sanitized = re.sub(r'\s{2,}', ' ', text_sanitized)
        text_sanitized = re.sub(r'^\s|\s$', '', text_sanitized)
        text_sanitized = text_sanitized.translate(
            str.maketrans('', '', string.punctuation)
        )
        return self.tokenizer(text_sanitized)
