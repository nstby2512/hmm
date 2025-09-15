import torch
from torchtext import data
from datasets.lm import WikiText2  # 或你改成 Childes 的 Dataset 类

# 定义文本字段
TEXT = data.Field()
batch_size = 32
bptt_len = 35

# 加载 Childes 数据集
train, val, test = WikiText2.iters(
    batch_size=batch_size,
    bptt_len=bptt_len,
    device=-1,  # CPU
    root='.data',
    train='childes.train',
    validation='childes.dev',
    test='childes.test'
)

print("训练集 batch 数量:", len(train))
print("验证集 batch 数量:", len(val))
print("测试集 batch 数量:", len(test))

# 检查每个 batch 的 token 数量
total_tokens = 0
for batch in train:
    # 假设 batch.text 的 shape: (seq_len, batch_size)
    total_tokens += batch.text.numel()

print("训练集总 token 数:", total_tokens)
