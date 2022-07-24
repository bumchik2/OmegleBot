import torch.nn as nn
import torch
import torch.nn.functional as F
import youtokentome as yttm
import os
from tqdm import tqdm
import numpy as np
import io
from nltk.tokenize import TweetTokenizer


vocab_size = 1000


def get_bpe(tokens, vocab_size=vocab_size):
    """
    Возвращает токенизатор BPE, обученный на токенах.
    Параметры.
    1) tokens - токены,
    2) vocab_size - количество уникальных токенов в итоговом словаре.
    """

    with open('tmp.json', 'w', encoding='utf-8') as file_:
        for token in tokens:
            print(token, file=file_)

    yttm.BPE.train('tmp.json', vocab_size=vocab_size, model='bpe.model')
    os.remove('tmp.json')

    return yttm.BPE(model="bpe.model")


with io.open('./nietzsche.txt', encoding='utf-8') as f:
    text = f.read().lower()

tokenizer = TweetTokenizer()
tokens = list(set(tokenizer.tokenize(text)))
bpe = get_bpe(tokens)

num_tokens = vocab_size


class TransformBasedModel(nn.Module):
    def __init__(self, device, num_tokens=num_tokens, emb_size=64, num_layers=7, dropout=0.15):
        super().__init__()

        self.emb = nn.Embedding(num_tokens, emb_size)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=8, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.hid_to_logits = nn.Linear(emb_size, num_tokens)
        self.device = device

    @staticmethod
    def generate_square_subsequent_mask(size):
        """ Функция генерации маски для предсказания, размера (seq_len, seq_len) """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # x: [batch_size, seq_len]
        # mask: [seq_len, seq_len]
        mask = self.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        emb = self.emb(x)  # batch_size x seq_len x emb_size
        output = self.transformer_encoder(emb, mask)  # batch_size x seq_len x emb_size
        output = self.hid_to_logits(output)
        logits = F.log_softmax(output, dim=-1)
        return logits


@torch.no_grad()
def beam_predict(model, text, length_to_predict, device, beam=10, beam_max=20):
    # beam - split at every step
    # beam_max - maximum beam width limit
    input_str_initial = bpe.encode(text, output_type=yttm.OutputType.SUBWORD, bos=True, eos=False)
    inputs = [bpe.subword_to_id(subword) for subword in input_str_initial]

    best_starts = [inputs]
    best_log_probs = [0]

    model.train(False)

    # at each step keep beam most probable sequences
    for _ in tqdm(range(length_to_predict)):
        possible_best_starts = []
        possible_best_log_probs = []

        for best_start, best_log_prob in zip(best_starts, best_log_probs):
            X_batch = torch.tensor(best_start).unsqueeze(dim=0).to(device)
            log_prob = model.forward(X_batch).cpu().numpy()  # [1, LENGTH, vocab_size]

            # look through the best beam tokens following the current best_start
            best_tokens = np.argsort(log_prob[0][-1])[-beam:]
            best_probs = log_prob[0][-1][best_tokens]

            for best_token, best_prob in zip(best_tokens, best_probs):
                possible_best_start = list(best_start).copy()
                possible_best_start.append(best_token)
                possible_best_starts.append(possible_best_start)
                possible_best_log_probs.append(best_log_prob + best_prob)

        # Choose the best min(beam_max, len(possible_best_starts)) starts
        top_number = min(beam_max, len(possible_best_starts))
        top_idx = np.argsort(possible_best_log_probs)[-top_number:]
        best_starts = np.array(possible_best_starts)[top_idx]
        best_log_probs = np.array(possible_best_log_probs)[top_idx]

    very_best_start = best_starts[np.argmax(best_log_probs)]

    # remove `end of string`
    eos_entrances = np.argwhere(very_best_start == bpe.subword_to_id('<EOS>'))[:, 0]
    if len(eos_entrances) > 0:
        very_best_start = very_best_start[:eos_entrances[0] + 1]
        best_log_probs = best_log_probs[:eos_entrances[0] + 1]

    # return best result
    best_result = torch.tensor(very_best_start[1:-1])
    return bpe.decode([list(best_result.numpy())])[0][len(text):]


def get_model(device, weights_path='nlp_transform_latest.pt'):
    model = TransformBasedModel(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.to(device)
    return model
