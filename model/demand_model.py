import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, hidden_size=128, output_size=1, num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        #self.dropout = nn.Dropout(dropout) 出力層にdropoutを適用する場合
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) # 最後の時間ステップを出力
        return out