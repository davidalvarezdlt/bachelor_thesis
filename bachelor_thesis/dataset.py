import bachelor_thesis.utils as utils
import torch.utils.data
import random
import torchtext
import torch.nn.utils.rnn


class BachelorThesisDataset(torch.utils.data.Dataset):
    data = None
    symbol = None
    years = None
    words_list = None
    n_news = None
    n_words = None
    items_indexes = None
    tokenizer = None

    def __init__(self, data, symbol, years, words_list, n_news, n_words):
        """Initializes the data set.

        The main task of the initialization is to fill ``self.items_indexes``,
        which contains the specific dates of the data associated to the data
        set. Only dates of the years given in ``years`` are valid.

        Args:
            data (h5py.File): data set containing both the news and the stock
                prices.
            symbol (str): symbol referencing the stock to be used.
            years (list): list containing the years to be used.
            words_list (list): ordered list contain the words of the
                dictionary.
            n_news (int): ``model/n_news`` configuration parameter.
            n_words (int): ``model/n_words`` configuration parameter.
        """
        self.data = data
        self.symbol = symbol
        self.years = years
        self.words_list = words_list
        self.n_news = n_news
        self.n_words = n_words
        self._init_items_indexes()
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

    def _init_items_indexes(self):
        self.items_indexes = []
        news_items_list = list(self.data['news/reuters'].keys())
        items_cands = filter(
            lambda x: int(x[:4]) in self.years,
            self.data['prices/{}'.format(self.symbol)].keys()
        )
        for item in items_cands:
            if item in news_items_list and len(
                    self.data['news/reuters'][item][()]) >= self.n_news:
                self.items_indexes.append(item)

    def __getitem__(self, item):
        """Returns the news and the movement to predict of a certain date.

        Returns a set of ``self.n_news`` news along with a binary value
        representing if the stock has increased or decreased its price during
        that date.

        Args:
            item (int): index of the date in ``self.items_indexes``.

        Returns:
            torch.Tensor: tensor of size (self.n_news, self.n_words) containing
                ``self.n_news`` random news of the date.
            int: 1 if the stock price has increased its price, 0 otherwise.
        """
        item_news = self.data['news/reuters'][self.items_indexes[item]][()]
        item_price = self.data[
            'prices/{}'.format(self.symbol)
        ][self.items_indexes[item]][()]
        news_titles = [
            item_news[new_index][0] for new_index in random.sample(
                list(range(item_news.shape[0])), self.n_news
            )
        ]
        return self._pack_news_titles(news_titles), \
            1 if item_price[3] > item_price[0] else 0

    def __len__(self):
        return len(self.items_indexes)

    def _pack_news_titles(self, news_titles):
        news_indexes = torch.zeros(
            (1, self.n_news, self.n_words), dtype=torch.int64
        )
        news_lengths = torch.zeros((1, self.n_news), dtype=torch.int64)
        for i, new_title in enumerate(news_titles):
            new_title_tokenized = utils.clean_and_tokenize(
                new_title.decode('utf-8'), self.tokenizer
            )[:self.n_words]
            for j, new_word in enumerate(new_title_tokenized):
                news_indexes[0, i, j] = self.words_list.index(new_word) \
                    if new_word in self.words_list else 1
            news_lengths[0, i] = len(new_title_tokenized)
        return news_indexes.squeeze(0), news_lengths.squeeze(0)
