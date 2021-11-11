"""
Module containing the ``pytorch_lightning.LightningModule`` implementation of
the package.
"""
import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F


class BachelorThesisModel(pl.LightningModule):
    """Implementation of the ``pytorch_lightning.LightningModule`` used in this
    package.

    Attributes:
        embd_layer: Instance of a ``torch.nn.Embedding`` layer.
        lstm_layer: Instance of a ``torch.nn.LSTM`` layer.
        linear_layer: Instance of a ``torch.nn.Linear`` layer.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """
    def __init__(self, embds_tensor, **kwargs):
        super(BachelorThesisModel, self).__init__()
        self.embd_layer = torch.nn.Embedding.from_pretrained(embds_tensor, 0)
        self.lstm_layer = torch.nn.LSTM(
            embds_tensor.shape[1], kwargs['lstm_hidden'], batch_first=True
        )
        self.linear_layer = torch.nn.Linear(kwargs['lstm_hidden'], 2)
        self.kwargs = kwargs

    def forward(self, x, x_length):
        """Forward pass through the model.

        Args:
            x: Tensor of size ``(--batch_size, --n_news, --max_words)``
                contaning the one-hot embedding of the first ``--max_words`` of
                the headlines of the iteration.
            x_length: Tensor of size ``(--batch_size, --n_news)``
                containing the real length of the headlines given in ``x``.

        Returns:
            Tensor of size ``(--batch_size)`` containing 1 if the prediction is
            that the stock price will increase its price, 0 if the prediction
            is that it will decrease.
        """
        x = self.embd_layer(x)

        news_encoded = []
        for i in range(x.size(1)):
            new_encoded, _ = self.lstm_layer(x[:, i])
            new_encoded_batch = []
            for b, new_length in enumerate(x_length[:, i]):
                new_encoded_batch.append(new_encoded[b, new_length - 1])
            news_encoded.append(torch.stack(new_encoded_batch))
        news_encoded = torch.stack(news_encoded, dim=1)

        news_encoded = F.dropout(news_encoded, p=self.kwargs['dropout_prob'])
        news_encoded_max, _ = news_encoded.max(dim=1)
        return self.linear_layer(news_encoded_max)

    def training_step(self, batch, batch_idx):
        """Performs a single pass through the training dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        x, x_length, y = batch
        y_hat = self(x, x_length)

        loss = F.cross_entropy(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / x.size(0)
        self.log('training_loss', loss)
        self.log('training_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single pass through the validation dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        x, x_lengths, y = batch
        y_hat = self(x, x_lengths)

        loss = F.cross_entropy(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / x.size(0)
        self.log('validation_loss', loss)
        self.log('validation_accuracy', accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single pass through the test dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        x, x_lengths, y = batch
        y_hat = self(x, x_lengths)

        loss = F.cross_entropy(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / x.size(0)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer used in the package.

        Returns:
            Instance of a configured ``torch.optim.Adam`` optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model-related CLI arguments to the parser.

        Args:
            parent_parser: Parser object just before adding the arguments.

        Returns:
            Parser object after adding the arguments.
        """
        parser = parent_parser.add_argument_group('BachelorThesisModel')
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lstm_hidden', type=int, default=1024)
        parser.add_argument('--dropout_prob', type=float, default=0.5)
        return parent_parser
