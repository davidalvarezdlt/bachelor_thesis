import torch
import torch.nn
import torch.nn.functional as F


class BachelorThesisToyModel(torch.nn.Module):

    def __init__(
            self, embds_tensor, embedding_size, lstm_hidden, dropout_prob
    ):
        super(BachelorThesisToyModel, self).__init__()
        self.embd_layer = torch.nn.Embedding.from_pretrained(
            embds_tensor, freeze=True, padding_idx=0
        )
        self.lstm_layer = torch.nn.LSTM(
            embedding_size, lstm_hidden, batch_first=True
        )
        self.linear_layer = torch.nn.Linear(lstm_hidden, 2)
        self.dropout_prob = dropout_prob

    def forward(self, news_indexes, news_lengths):
        x = self.embd_layer(news_indexes)

        # Apply the LSTM encoder to each new, extract the encoding at the
        # position of the last word
        news_encoded = []
        for i in range(x.size(1)):
            new_encoded, _ = self.lstm_layer(x[:, i])
            new_encoded_batch = []
            for b, new_length in enumerate(news_lengths[:, i]):
                new_encoded_batch.append(new_encoded[b, new_length - 1])
            news_encoded.append(torch.stack(new_encoded_batch))
        news_encoded = torch.stack(news_encoded, dim=1)

        # Apply Dropout and Max-Pooling over the hidden features
        news_encoded = F.dropout(news_encoded, p=self.dropout_prob)
        news_encoded_max, _ = news_encoded.max(dim=1)

        # Apply the linear layer
        return self.linear_layer(news_encoded_max)
