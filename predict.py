import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pickle

def predict(vfilename, wfilename, filename):
    with open(vfilename, 'rb') as f:
        (token2int, int2token) = pickle.load(f)

    block_size=10
    n_outputs = 2
    vocab_size = len(token2int) + 2

    from mingpt.model import GPT, GPTConfig
    mconf = GPTConfig(vocab_size, n_outputs, block_size, n_layer=1, n_head=1, n_embd=32) # toy
    model = GPT(mconf)

    model.load_state_dict(torch.load(wfilename))
    model.eval()

    from mingpt.utils import predict

    def tok2int(token):
        if token in token2int:
            return token2int[token]
        else:
            i = len(token2int) + 2
            token2int[token] = i
            int2token[i] = token
            return i

    with open(filename + ".predict", "w") as out:
        with open(filename, "r") as f:
            for line in f.readlines():
                tokens    = line.strip().split(" ")
                ints      = [tok2int(token) for token in tokens]

                x_in      = torch.tensor([ints + [1]], dtype=torch.long)
                guesses   = predict(model, x_in)
                out.write(str(float(guesses[0, 0])))
                out.write("\n")

if __name__ == "__main__":
    predict("dummy.out.vocab.pkl", "dummy.out.trained", "dummy.predict")
