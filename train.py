import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pickle

def train(filename):
    ## Super naive, just to get something off the ground
    ## 0 is left-pad, 1 is EOI, tokens start at 2
    token2int  = {}
    int2token  = {}

    datapoints = []

    maxTokens  = -1

    def tok2int(token):
        if token in token2int:
            return token2int[token]
        else:
            i = len(token2int) + 2
            token2int[token] = i
            int2token[i] = token
            return i

    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens    = line.strip().split(" ")
            label     = int(tokens[0])
            tokens    = [tok2int(token) for token in tokens[1:]]
            maxTokens = max(maxTokens, len(tokens))
            datapoints.append((tokens, label))

    inputs  = torch.zeros((len(datapoints), maxTokens + 1), dtype=torch.long)
    outputs = torch.zeros((len(datapoints), 1), dtype=torch.long)

    for i, (tokens, label) in enumerate(datapoints):
        nPad = maxTokens - len(tokens)
        inputs[i, :] = torch.tensor([0 for _ in range(nPad)] + tokens + [1], dtype=torch.long)
        outputs[i]   = torch.tensor([label], dtype=torch.long)

    with open(filename + ".vocab.pkl", 'wb') as f:
        pickle.dump((token2int, int2token), f)

    dataset = TensorDataset(inputs, outputs)

    block_size=10
    n_outputs = 2
    vocab_size = len(token2int) + 2

    from mingpt.model import GPT, GPTConfig
    mconf = GPTConfig(vocab_size, n_outputs, block_size, n_layer=1, n_head=1, n_embd=32) # toy
    model = GPT(mconf)

    from mingpt.trainer import Trainer, TrainerConfig
    tconf = TrainerConfig(max_epochs=1000, batch_size=256)
    trainer = Trainer(model, dataset, dataset, tconf)
    trainer.train()

    torch.save(model.state_dict(), filename + ".trained")

if __name__ == "__main__":
    train("dummy.out")
