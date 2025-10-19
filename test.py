import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt # 用於可視化

# --- 1. 準備數據 ---
# 我們希望模型學習 y = 2x + 1
# 創建 100 個 x 數據點 (作為輸入特徵)
# .unsqueeze(1) 是為了把 (100,) 變成 (100, 1)，符合 nn.Linear 的輸入要求
X_train = torch.randn(100, 1) 

# 根據 x 創建 y (作為真實標籤)，並加入一點點噪聲
# 這樣模型才需要去 "學習" 而不是 "背誦"
y_train = 2 * X_train + 1 + 0.1 * torch.randn(100, 1)

# 畫出原始數據看看
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Data")
plt.scatter(X_train.numpy(), y_train.numpy())
plt.xlabel("x")
plt.ylabel("y")


# --- 2. 定義模型 ---
# 我們的模型非常簡單，只有一個線性層
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        # 輸入特徵是 1 (x)，輸出特徵也是 1 (y)
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # 直接將 x 傳遞給線性層
        return self.linear(x)

# 實例化模型
model = SimpleLinearRegression()
print(f"模型結構: {model}")


# --- 3. 定義損失函數和優化器 ---
# 這是一個回歸問題，所以使用均方誤差 (Mean Squared Error)
criterion = nn.MSELoss()

# 使用隨機梯度下降(SGD)作為優化器，告訴它去更新 model.parameters()
# lr 是學習率 (learning rate)，控制每一步更新的幅度
optimizer = optim.SGD(model.parameters(), lr=0.01)


# --- 4. 訓練模型 ---
num_epochs = 200 # 總共訓練 200 輪
losses = [] # 用來記錄每一輪的 loss

print("\n--- 開始訓練 ---")
for epoch in range(num_epochs):
    # 1. 前向傳播：獲得預測值
    y_predicted = model(X_train)
    
    # 2. 計算損失
    loss = criterion(y_predicted, y_train)
    losses.append(loss.item())
    
    # 3. 反向傳播：計算梯度
    #    在 backward 之前必須先清空之前的梯度
    optimizer.zero_grad()
    loss.backward()
    
    # 4. 更新參數
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("--- 訓練完成 ---")


# --- 5. 查看結果 ---
# 提取學習到的參數
# .item() 是為了從張量中取出數值
learned_weight = model.linear.weight.item()
learned_bias = model.linear.bias.item()

print(f"\n真實模型: y = 2.0x + 1.0")
print(f"學到模型: y = {learned_weight:.4f}x + {learned_bias:.4f}")

# 畫出模型學到的線
plt.subplot(1, 2, 2)
plt.title("Trained Model Fit")
plt.scatter(X_train.numpy(), y_train.numpy(), label="Original Data")
# 用模型對原始 x 進行預測，畫出這條線
predicted = model(X_train).detach().numpy() # .detach() 停止追蹤梯度
plt.plot(X_train.numpy(), predicted, 'r', label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

# 畫出 Loss 下降曲線
plt.figure()
plt.title("Training Loss")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()