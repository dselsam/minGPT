import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pickle

def predict(vocab, checkpoint, infile, outfile):
    with open(vocab, 'rb') as f:
        (token2int, int2token) = pickle.load(f)

    block_size=1000
    n_outputs = 2
    vocab_size = len(token2int) + 2

    from mingpt.model import GPT, GPTConfig
    mconf = GPTConfig(vocab_size, n_outputs, block_size, n_layer=1, n_head=1, n_embd=32) # toy
    model = GPT(mconf)

    model.load_state_dict(torch.load(checkpoint))
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

    with open(outfile, "w") as out:
        with open(infile, "r") as f:
            for line in f.readlines():
                tokens    = line.strip().split(" ")
                ints      = [tok2int(token) for token in tokens]

                x_in      = torch.tensor([ints + [1]], dtype=torch.long)
                guesses   = predict(model, x_in)
                out.write(str(float(guesses[0, 0])))
                out.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", action="store", dest='infile', type=str, default='predict-in')
    parser.add_argument("--outfile", action="store", dest='outfile', type=str, default='predict-out')
    parser.add_argument("--vocab", action="store", dest='vocab', type=str, default='vocab.pkl')
    parser.add_argument("--checkpoint", action="store", dest='checkpoint', type=str, default='checkpoints/default')

    opts = parser.parse_args()

    predict(vocab=opts.vocab,
            checkpoint=opts.checkpoint,
            infile=opts.infile,
            outfile=opts.outfile)
