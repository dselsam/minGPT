import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

inputs = torch.tensor([
    [0, 3, 4, 5, 6, 7, 1],
    [0, 7, 6, 5, 4, 3, 1],
    [3, 4, 5, 6, 7, 7, 1],
    [7, 7, 6, 5, 4, 3, 1]
    ])

outputs = torch.tensor([[0], [1], [0], [1]])

print(inputs.dtype, outputs.dtype)

train_dataset = TensorDataset(inputs, outputs)

test_dataset = train_dataset

vocab_size=10

# TODO: for now binary classification, until we have fancier models
# worse, we label non-ultimate tokens with 2 indicating not ready
n_outputs=2
block_size=10

# construct a GPT model
from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(vocab_size, n_outputs, block_size, n_layer=1, n_head=1, n_embd=32) # toy
model = GPT(mconf)

# construct a trainer
from mingpt.trainer import Trainer, TrainerConfig
tconf = TrainerConfig(max_epochs=1000, batch_size=256)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()
# (... enjoy the show for a while... )

# sample from the model (the [None, ...] and [0] are to push/pop a needed dummy batch dimension)
from mingpt.utils import predict
print("[good] 0: ", predict(model, torch.tensor([[0, 3, 4, 5, 6, 7, 1]], dtype=torch.long)))
print("[good] 1: ", predict(model, torch.tensor([[0, 7, 6, 5, 4, 3, 1]], dtype=torch.long)))

print("[ok  ] 0: ", predict(model, torch.tensor([[0, 0, 3, 4, 5, 6, 1]], dtype=torch.long)))
print("[good] 1: ", predict(model, torch.tensor([[0, 0, 6, 5, 4, 3, 1]], dtype=torch.long)))
