import torch
from torch import nn
from vocab import Vocab

class PyTorchTransformerEncoderModel(nn.Module):
    def __init__(self, d_model: int, head: int, n_layer: int, d_ff: int, dropout: float, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.total_tokens, d_model)

        # Khởi tạo TransformerEncoderLayer với batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=head, 
            dim_feedforward=d_ff, 
            dropout=dropout, 
            activation="relu",
            batch_first=True  # Quan trọng: Cho phép sử dụng batch_size ở trục đầu tiên
        )

        # Tạo stack 3 lớp encoder
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.cls_token = vocab.cls_idx
        self.lm_head = nn.Linear(d_model, vocab.total_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        # Tạo attention mask (0 là pad, 1 là valid token)
        attention_mask = input_ids != self.vocab.pad_idx

        # Embedding đầu vào
        input_embs = self.embedding(input_ids)  # (batch_size, seq_len, d_model)

        # Transformer Encoder
        features = self.encoder(input_embs, src_key_padding_mask=~attention_mask)  # Ngược mask: True là pad

        # Sử dụng token <cls> để phân loại
        cls_features = features[:, 0, :]  # (batch_size, d_model)

        # Linear Head
        logits = self.dropout(self.lm_head(cls_features))  # (batch_size, num_labels)

        # Tính loss
        loss = self.loss_fn(logits, labels.view(-1))  # Đảm bảo nhãn có đúng kích thước
        return logits, loss
