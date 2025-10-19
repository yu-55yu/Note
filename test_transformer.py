import torch
import torch.nn as nn
import torch.optim as optim
import math

# -----------------------------------------------------------------
# 1. 位置編碼 (Positional Encoding)
# 這是唯一一個你需要從官方教程 "複製" 過來的類
# 它的作用是為輸入向量添加關於 "位置" 的信息
# -----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 創建一個長為 max_len, 寬為 d_model 的位置編碼矩陣
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # register_buffer 會將 pe 註冊為模型的一部分 (不是可訓練參數)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 的形狀: (seq_len, batch_size, d_model)
        """
        # 將 x 與其對應的位置編碼相加
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

# -----------------------------------------------------------------
# 2. 定義我們的模型 (這就是你唯一需要寫的類)
# -----------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 d_hid: int, nlayers: int, num_classes: int, 
                 dropout: float = 0.5):
        """
        參數:
        vocab_size: 詞彙表示 (我們的例子裡是 0-9 總共 10 個數字)
        d_model: 模型的 "隱藏層維度" (embedding dimension)
        nhead: 多頭注意力的 "頭數"
        d_hid: 前饋網絡 (FeedForward) 的隱藏層維度
        nlayers: Transformer Encoder Layer 的層數
        num_classes: 最終輸出的分類數 (奇數/偶數，共 2 類)
        """
        super().__init__()
        self.d_model = d_model
        
        # 1. 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置編碼 (使用我們上面複製的類)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. 官方 Transformer Encoder Layer
        #    我們在這裡設置 batch_first=True, 這樣輸入維度就是 (N, S, E)
        #    N = batch_size, S = sequence_length, E = d_model
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        
        # 4. 官方 Transformer Encoder (由多個 Layer 堆疊而成)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # 5. 最終的分類器 (一個線性層)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src 的形狀: (batch_size, seq_len)
        """
        # 1. 嵌入: (N, S) -> (N, S, E)
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # 2. 位置編碼
        #    因為 PositionalEncoding 默認是 (S, N, E), 
        #    而我們用了 batch_first=True, 所以需要調整一下維度
        #    (N, S, E) -> (S, N, E)
        src = src.transpose(0, 1) 
        src = self.pos_encoder(src)
        # 轉回來: (S, N, E) -> (N, S, E)
        src = src.transpose(0, 1)
        
        # 3. Transformer 編碼: (N, S, E) -> (N, S, E)
        #    這裡我們不需要 mask
        output = self.transformer_encoder(src)
        
        # 4. 分類
        #    我們只取第一個 token (通常是 [CLS] token) 的輸出來做分類
        #    (N, S, E) -> (N, E)
        output = output[:, 0, :]
        
        # 5. 線性層: (N, E) -> (N, num_classes)
        output = self.classifier(output)
        return output
    
# -----------------------------------------------------------------
# 3. 完整的訓練 Demo
# -----------------------------------------------------------------

# --- 1. 定義超參數 ---
VOCAB_SIZE = 10     # 詞彙表大小 (數字 0-9)
D_MODEL = 32        # Embedding 維度
NHEAD = 4           # 頭數 (必須能被 D_MODEL 整除)
D_HID = 64          # 前饋網絡隱藏層維度
NLAYERS = 2         # Encoder 層數
NUM_CLASSES = 2     # 輸出分類數 (奇數/偶數)
DROPOUT = 0.1

SEQ_LEN = 10        # 序列固定長度
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# --- 2. 創建模型、損失函數和優化器 ---
model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NHEAD, D_HID, NLAYERS, NUM_CLASSES, DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("--- 模型結構 ---")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"總可訓練參數: {total_params}")


# --- 3. 訓練迴圈 ---
print("\n--- 開始訓練 ---")
model.train() # 設置為訓練模式

for epoch in range(NUM_EPOCHS):
    # --- 創建假的訓練數據 ---
    # 創建 (BATCH_SIZE, SEQ_LEN) 的隨機數字 (0-9)
    src_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # 創建標籤：計算每一行的總和，判斷奇偶
    # (BATCH_SIZE,)
    labels = src_data.sum(dim=1) % 2
    
    # --- 訓練步驟 ---
    # 1. 清零梯度
    optimizer.zero_grad()
    
    # 2. 前向傳播
    # 這裡我們沒有使用 padding mask, 因為所有序列長度都一樣
    output = model(src_data)
    
    # 3. 計算損失
    loss = criterion(output, labels)
    
    # 4. 反向傳播
    loss.backward()
    
    # 5. 更新參數
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        # 計算準確率
        preds = output.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        print(f'Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

print("--- 訓練完成 ---")


# --- 4. 測試模型 ---
model.eval() # 設置為評估模式
with torch.no_grad(): # 關閉梯度計算
    # 創建一筆測試數據
    test_data = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]]) # 總和 15 (奇數)
    test_label = test_data.sum(dim=1) % 2
    
    print(f"\n測試數據: {test_data}")
    print(f"真實標籤: {test_label.item()} (0=偶, 1=奇)")
    
    pred_output = model(test_data)
    pred_class = pred_output.argmax(dim=1)
    
    print(f"模型預測: {pred_class.item()} (0=偶, 1=奇)")

    # 創建另一筆測試數據
    test_data = torch.tensor([[2, 4, 6, 8, 0, 0, 0, 0, 0, 0]]) # 總和 20 (偶數)
    test_label = test_data.sum(dim=1) % 2
    
    print(f"\n測試數據: {test_data}")
    print(f"真實標籤: {test_label.item()} (0=偶, 1=奇)")
    
    pred_output = model(test_data)
    pred_class = pred_output.argmax(dim=1)
    
    print(f"模型預測: {pred_class.item()} (0=偶, 1=奇)")