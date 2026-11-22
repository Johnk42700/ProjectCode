"""The first step trains one speaker-listener pair in the reconstruction game.
The speaker describes a training object with a message and learns with
REINFORCE; the listener reconstructs the object from the message and learns
with supervised cross-entropy. Accuracy on the 52 held-out test objects, which
the agents never see during training, measures generalisation.

Structure of this file:
Config          all hyper-parameters of one run
Run             agents + optimisers + one game step + evaluation
train_pair      the training loop: play steps games, evaluate every few steps
run_experiment  builds a Run, trains it, returns a JSON-able dict """

import time
from dataclasses import dataclass, asdict
import numpy as np
import torch
from agents import Listener, Speaker
from game import ObjectSpace, attribute_nll

@dataclass
class Config:
    seed: int = 0
    # game: 4 attributes x 4 values, vocabulary 20, messages of length 10 
    n_attributes: int = 4
    n_values: int = 4
    vocab_size: int = 20
    msg_len: int = 10
    # agents and optimisation
    hidden: int = 64
    lr_speaker: float = 2e-3
    lr_listener: float = 2e-3
    batch: int = 256
    entropy_coef: float = 0.02
    obj_input: bool = True            # speaker sees the object at every step
    # schedule
    steps: int = 3000                 # game steps
    # evaluation
    n_evals: int = 24                 # checkpoints along training
    progress: str = ""                

    
class Run:
    """Holds the agents, optimisers and logging of one training run."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Seeding the object split and the agents' initial weights which are solely dependent on the seed value
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.space = ObjectSpace(cfg.n_attributes, cfg.n_values, seed=cfg.seed)
        self.gen = torch.Generator().manual_seed(1000 + cfg.seed)   # batch sampling
        self.n_in = cfg.n_attributes * cfg.n_values

        # lists with one agent each;
        self.speakers = [self.new_speaker()]
        self.listeners = [self.new_listener()]

        # every agent has its own Adam optimiser (its own moment estimates)
        self.opt_s = [self.new_opt(s, cfg.lr_speaker) for s in self.speakers]
        self.opt_l = [self.new_opt(l, cfg.lr_listener) for l in self.listeners]
        self.trajectory = []          # one dict per checkpoint, see checkpoint()
        self.speaker_updates = 0
        self.listener_updates = 0
        self.t0 = time.time()

    def new_speaker(self):
        conf = self.cfg
        return Speaker(self.n_in, conf.vocab_size, conf.msg_len, conf.hidden, obj_input=conf.obj_input)

    def new_listener(self):
        conf = self.cfg
        return Listener(conf.n_attributes, conf.n_values, conf.vocab_size, conf.hidden)

    @staticmethod
    def new_opt(module, lr):
        return torch.optim.Adam(module.parameters(), lr)

    # ------------------------------------------------------------------ steps
    def game_step(self, s, l, update_s=True, update_l=True):
        """One game on a batch of training objects; optionally update each agent.
        Speaker s describes a batch of training objects, listener l decodes the messages. Then:
          listener: one supervised gradient step on the reconstruction loss
          speaker: one REINFORCE step. Messages are discrete, so no gradient
            flows from the listener into the speaker instead the speaker makes messages that earned a high reward more likely:
                loss_S = -mean(advantage * log pi(m|x)) - entropy_coef * entropy
            with reward r = mean over attributes of log rho_k(x_k|m) (the listener's log-likelihood) and advantage = r normalised to zero mean
            and unit variance within the batch to reduce variance. """
        c = self.cfg
        S, L = self.speakers[s], self.listeners[l]
        idx = self.space.sample_train(c.batch, self.gen)
        x = self.space.onehot(idx)
        obj = self.space.objects[idx]
        if update_s:
            msgs, logp, ent = S(x)
        else:
            with torch.no_grad():               # speaker frozen: no graph needed
                msgs, logp, ent = S(x)
        nll = attribute_nll(L(msgs), obj)       # -log rho(x|m) per example
        if update_l:
            self.opt_l[l].zero_grad()
            nll.mean().backward()
            self.opt_l[l].step()
            self.listener_updates += 1
        if update_s:
            
            # reward computed with the listener before its update in this step
            # detach() so the speaker loss does not backpropagate into the listener
            r = -nll.detach() / c.n_attributes
            adv = (r - r.mean()) / (r.std() + 1e-8)
            loss = -(adv * logp).mean() - c.entropy_coef * (ent / c.msg_len).mean()
            self.opt_s[s].zero_grad()
            loss.backward()
            self.opt_s[s].step()
            self.speaker_updates += 1

# ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def accuracy(self, S, L):
        """Fraction of train / test objects whose 4 attributes are ALL decoded correctly (greedy)."""
        allidx = torch.arange(self.space.n_objects)
        msgs, _, _ = S(self.space.onehot(allidx), greedy=True)
        correct = (L(msgs).argmax(-1) == self.space.objects).all(-1).float()
        return float(correct[self.space.train_idx].mean()), float(correct[self.space.test_idx].mean())

    @torch.no_grad()
    def greedy_messages(self, S):
        """The speaker's deterministic language: its most likely message for each of the 256 objects."""
        msgs, _, _ = S(self.space.onehot(torch.arange(self.space.n_objects)), greedy=True)
        return msgs.cpu().numpy()

    def checkpoint(self, step):
        """Accuracy of the pair at this step"""
        
        S, L = self.speakers[0], self.listeners[0]
        train_acc, test_acc = self.accuracy(S, L)
        rec = {"step": step, "train_acc": train_acc, "test_acc": test_acc,
               "speaker_updates": self.speaker_updates, "listener_updates": self.listener_updates}
        self.trajectory.append(rec)
        if self.cfg.progress and step >= 0:
            print(f"  .. {self.cfg.progress}: step {step}/{self.cfg.steps}  train acc {train_acc:.2f}  "
                  f"test acc {test_acc:.2f}  ({(time.time() - self.t0) / 60:.1f} min)", flush=True)
        return rec

    def final_eval(self):
        """End-of-training metrics: accuracy and how many distinct messages the speaker uses."""
        last = self.checkpoint(-1)          # step -1 marks "final"
        m = self.greedy_messages(self.speakers[0])
        return {"train_acc": last["train_acc"], "test_acc": last["test_acc"],
                "n_unique_msgs": int(len({tuple(r) for r in m}))}


# ---------------------------------------------------------------- training loop
def _eval_steps(total, n_evals):
    """The n_evals equally spaced steps (1..total) at which checkpoints are taken."""
    every = max(1, total // n_evals)
    return set(range(every, total + 1, every))


def train_pair(run: Run):
    """The standard pair: speaker i and listener j play steps games, both learn from every game."""
    c = run.cfg
    evals = _eval_steps(c.steps, c.n_evals)
    for step in range(1, c.steps + 1):
        run.game_step(0, 0)
        if step in evals:
            run.checkpoint(step)


def run_experiment(cfg: Config):
    """Run one configuration end to end and return everything that would be saved to JSON."""
    t0 = time.time()
    run = Run(cfg)
    train_pair(run)
    final = run.final_eval()
    traj = run.trajectory[:-1]  # last entry is the final checkpoint
    return {
        "config": asdict(cfg),
        "final": final,
        "trajectory": traj,
        "runtime_sec": time.time() - t0,
    }
