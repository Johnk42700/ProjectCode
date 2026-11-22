"""Speaker and listener networks (LSTM agents as in Rita et al., 2022a/b).

Messages are fixed-length sequences of T symbols from a vocabulary of size W. Both agents use a single-layer LSTM; the speaker
LSTM is initialised from a linear embedding of the object, the listener reads
the message and predicts each attribute with an independent softmax head.


Data flow between agents
---------
Speaker  (object -> message):

one-hot object x --Linear--> h_0 (initial LSTM state)

step t:  [embed(previous symbol) ; embed(x)] --LSTM--> o_t --Linear--> logits over W symbols
         sample m_t ~ softmax(logits)   (or argmax when greedy)
message m = (m_1, ..., m_T),  log pi(m|x) = sum_t log pi(m_t | m_<t, x)

---------

Listener (message -> object):

m_1...m_T --embed--> LSTM --last hidden state--> Linear --> K x V logits
rho(x|m) (Listener policy) = prod_k softmax(logits_k)[x_k]     (attributes predicted independently)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Speaker(nn.Module):
    def __init__(self, n_inputs, vocab_size=20, msg_len=10, hidden=128, emb=32, obj_input=False):
        super().__init__()
        self.W= vocab_size
        self.T= msg_len
        self.H= hidden

        self.obj_input = obj_input
        self.obj_to_hidden = nn.Linear(n_inputs, hidden)

        self.embed = nn.Embedding(vocab_size + 1, emb)
        self.obj_emb = nn.Linear(n_inputs, emb) if obj_input else None
        self.lstm = nn.LSTM(2 * emb if obj_input else emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)  # hidden state -> logits over the W symbols

    def _init_state(self, x):
            """Initial (h, c) of the LSTM: h from the object, c = 0. Shapes [1, B, H]."""
            h0 = self.obj_to_hidden(x).unsqueeze(0)
            return h0, torch.zeros_like(h0)
    
    def _inputs(self, x, tokens):
            """LSTM inputs for a sequence of previous symbols: [B, L] -> [B, L, emb(+emb)]."""
            emb = self.embed(tokens)
            if self.obj_input:
                o = self.obj_emb(x).unsqueeze(1).expand(-1, tokens.shape[1], -1)
                emb = torch.cat([emb, o], -1)
            return emb
    def forward(self, x, greedy=False):
            """Sample (or greedily decode) messages.
            Returns messages [B, T], summed log-probs [B] and summed per-step policy entropies [B]. """
            B = x.shape[0]
            
            state = self._init_state(x)
    
            tok = torch.full((B, 1), self.W, dtype=torch.long, device=x.device)  # start symbol
    
            msgs, logps, ents = [], [], []
            for _ in range(self.T):
                # one LSTM step: feed the previous symbol, keep the recurrent state
                o, state = self.lstm(self._inputs(x, tok), state)
                logits = self.out(o[:, 0])
                dist = torch.distributions.Categorical(logits=logits)
                s = logits.argmax(-1) if greedy else dist.sample()
                msgs.append(s)
                logps.append(dist.log_prob(s))   # log pi(m_t | m_<t, x), keeps the graph for REINFORCE
                ents.append(dist.entropy())      # entropy of this step's distribution (exploration bonus)
                tok = s.unsqueeze(1)             # the chosen symbol is the next input
            # log pi(m|x) is the sum over steps because pi(m|x) = prod_t pi(m_t | m_<t, x)
            return torch.stack(msgs, 1), torch.stack(logps, 1).sum(1), torch.stack(ents, 1).sum(1)
    
    def log_prob(self, x, msgs):
            """Teacher-forced log pi(m|x) for given messages; x [B, n_in], msgs [B, T]."""
            B = x.shape[0]
            # inputs are start (start symbol), m_1, ..., m_{T-1}; the outputs predict m_1, ..., m_T
            start = torch.full((B, 1), self.W, dtype=torch.long, device=x.device)
            inp = torch.cat([start, msgs[:, :-1]], 1)
            o, _ = self.lstm(self._inputs(x, inp), self._init_state(x))  # whole sequence in one call
            logp = F.log_softmax(self.out(o), -1)                        # [B, T, W]
            # log-probability of the symbol actually in the message at every step, summed over steps
            return logp.gather(-1, msgs.unsqueeze(-1)).squeeze(-1).sum(-1)
    
    
class Listener(nn.Module):
        def __init__(self, n_attributes, n_values, vocab_size=20, hidden=128, emb=32):
            super().__init__()
            self.K, self.V = n_attributes, n_values
            self.embed = nn.Embedding(vocab_size, emb)
            self.lstm = nn.LSTM(emb, hidden, batch_first=True)
            # one linear layer producing all K*V logits, reshaped into K separate softmax heads
            self.heads = nn.Linear(hidden, n_attributes * n_values)
    
        def forward(self, msgs):
            """Logits [B, K, V] for each attribute given messages [B, T]."""
            _, (h, _) = self.lstm(self.embed(msgs))   # h: final hidden state after reading all T symbols
            return self.heads(h[-1]).view(-1, self.K, self.V)