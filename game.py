"""Lewis reconstruction game: object space, train/test split and losses.

Objects have K attributes with V values each (default 4 x 4 = 256 objects,
as in Rita et al., 2022a). An object is stored as a LongTensor of K attribute
values; the speaker sees the concatenation of K one-hot vectors.


How the game works
------------------
1. An object x = (x_1, ..., x_K) is drawn from the training set.
2. The speaker turns x into a message m (see agents.Speaker).
3. The listener reads m and outputs, for every attribute k, a probability
   distribution rho_k(. | m) over the V possible values (see agents.Listener).
4. Both agents are scored by how much probability the listener puts on the
   true object: rho(x | m) = prod_k rho_k(x_k | m). The loss is its negative
   logarithm, computed by attribute_nll below.

Everything about the world lives in ObjectSpace: the list of all objects,
the train/test split, and helpers to sample and encode objects. Because the
space is small (256 objects) we can enumerate it completely.
"""


import itertools
import torch
import torch.nn.functional as F

class ObjectSpace:
    def __init__(self, n_attributes=4, n_values=4, train_frac=0.8, seed=0, device="cpu"):
        self.K = n_attributes
        self.V = n_values

        # All V**K objects in a fixed (lexicographic) order: (0,0,0,0), (0,0,0,1), ...
        # An object is referred to everywhere by its row index in this table.

        values = list(itertools.product(range(n_values), repeat=n_attributes))
        self.objects = torch.tensor(values, dtype=torch.long, device="cpu")  # [n_objects, K]
        self.n_objects = len(values)

        # Random 80/20 split that depends only on the seed
        # 204 training and 52 test objects.
        # Test objects are never shown to the agents during training they measure generalisation to unseen attribute combinations.
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(self.n_objects, generator=g, device="cpu")
        n_train = int(train_frac * self.n_objects)
        self.train_idx = perm[:n_train].sort().values.to(device)
        self.test_idx = perm[n_train:].sort().values.to(device)
        self.objects = self.objects.to(device)

    def onehot(self, idx):
            """One-hot encoding [B, K*V] of the objects with indices idx.
            Example with K=2, V=3: object (2, 0) -> [0,0,1, 1,0,0]. This vector is the speaker's input. """
            obj = self.objects[idx]
            return F.one_hot(obj, self.V).float().view(len(idx), -1)

    def sample_train(self, batch_size, gen=None):
            """Indices of batch_size training objects drawn uniformly with replacement.
            The training set has 204 objects, so a batch of 256 contains repeats."""
            
            pos = torch.randint(len(self.train_idx), (batch_size,), generator=gen)
            return self.train_idx[pos]

def attribute_nll(listener_logits, objects):
    """Per-example joint negative log-likelihood -log rho(x|m) = -sum_k log rho_k(x_k|m).
    listener_logits: [B, K, V]; objects: [B, K]. Returns [B].
    This one quantity plays three roles in the code:
      it is the listener's supervised training loss (cross-entropy summed over
        the K attributes),
      its negative (divided by K) is the speaker's REINFORCE reward,
      its average is the "listener loss" that the diagnostic splits into an
        information part and a co-adaptation part. """
    logp = F.log_softmax(listener_logits, dim=-1)  # log rho_k(v | m) for all k, v pick, for every attribute, the log-probability of the true value and sum over attributes
    return -logp.gather(-1, objects.unsqueeze(-1)).squeeze(-1).sum(-1)