import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerBiLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 num_classes, 
                 d_model=64, 
                 nhead=8, 
                 num_transformer_layers=2, 
                 lstm_hidden_size=64, 
                 num_lstm_layers=1, 
                 dropout=0.2):
        """
        参数说明：
        - input_size: 输入特征的维度 (例如：如果是单通道的传感器序列，通常为1；如果有多个传感器，则等于传感器数量)
        - num_classes: 分类的类别数 (故障类型数量)
        - d_model: Transformer内部特征维度
        - nhead: Transformer的多头注意力头数 (注意：d_model必须能被nhead整除)
        - num_transformer_layers: Transformer编码器的层数
        - lstm_hidden_size: BiLSTM 的隐藏层维度
        - num_lstm_layers: BiLSTM 的层数
        - dropout: 随机失活概率，用于防止过拟合
        """
        super(TransformerBiLSTM, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # BiLSTM 层
        self.bilstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=lstm_hidden_size, 
            num_layers=num_lstm_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)  # shape: (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)       # shape: (batch_size, seq_len, d_model)
        
        # shape 保持不变: (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)
        
        # lstm_out shape: (batch_size, seq_len, 2 * lstm_hidden_size)
        # h_n shape: (2 * num_lstm_layers, batch_size, lstm_hidden_size)
        lstm_out, (h_n, c_n) = self.bilstm(x)

        last_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) 
        # last_hidden_state shape: (batch_size, 2 * lstm_hidden_size)
        
        # 最终分类
        out = self.fc(last_hidden_state) # shape: (batch_size, num_classes)
        
        return out



if __name__ == "__main__":
    # 数据集参数
    BATCH_SIZE = 32
    SEQ_LEN = 1024       # 时间序列长度
    INPUT_SIZE = 1       # 每个时间步的特征维度
    NUM_CLASSES = 10     # 类型数
    
    model = TransformerBiLSTM(
        input_size=INPUT_SIZE, 
        num_classes=NUM_CLASSES, 
        d_model=64,             # 注意这里被 Transformer 映射
        nhead=8,                # d_model需被nhead整除
        num_transformer_layers=2, 
        lstm_hidden_size=64, 
        num_lstm_layers=1
    )
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"当前设备: {device}")
    
    # 这里随机生成一批虚拟数据来测试
    # 形状必须满足：(Batch_Size, Sequence_Length, Features_Size)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE).to(device)
    
    print(f"输入数据的Shape: {dummy_input.shape} => (batch_size, seq_len, input_size)")
    
    output = model(dummy_input)
    
    print(f"输出数据的Shape: {output.shape} => (batch_size, num_classes)")
    print("模型可以正常工作！后期直接将真实的 DataLoader 传入并加入普通的交叉熵损失(CrossEntropyLoss)进行训练即可。")