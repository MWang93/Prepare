import torch

# 序列长度seq_len=5, batch_size=3, 数据向量维数=10
# 长度10的向量 * 3 = 1个batch, 这个tensor有5个batch [有5组数据，每组3行，每行10列]
input = torch.randn(5, 3, 10)
print(input.size(2))