import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator

# Construct a confusion matrix for binary classification.
def confusion_matrix(pred: Tensor, targ: Tensor):
  """
  pred, targ: Tensors of length n
  pred: float in [0,1] = stress probability
  targ: 0 (calm, negative) or 1 (stress, positive)
  """
  pred = torch.round(pred)
  tp = torch.sum(pred * targ)
  fp = torch.sum(pred * (1-targ))
  fn = torch.sum((1-pred) * targ)
  tn = torch.sum((1-pred) * (1-targ))
  # Row is target, column is prediction. (0:calm, 1:stress)
  return Tensor([[tn, fp], [fn, tp]])

# Compute prediction qualities from a tensor of confusion matrices
@dataclass
class PredictionMetrics:
  accuracy: float # or Tensor theoreof
  precision: float
  sensitivity: float
  specificity: float
  f1: float

  @classmethod
  def from_confusion(cls, c: Tensor):
    # c=confusion: shape (..., 2, 2), where last two dims are confusion matrix.
    # Preceding dims are preserved
    tn, fp, fn, tp = c[..., 0, 0], c[..., 0, 1], c[..., 1, 0], c[..., 1, 1]
    if tn.numel() == 1:
      tn, fp, fn, tp = tn.item(), fp.item(), fn.item(), tp.item()

    #d = c.dim()-2
    #tn, fp, fn, tp = list(c.flatten(start_dim=-2).permute((d, *range(d))))

    return cls(
      accuracy = (tp + tn) / (tp + tn + fp + fn),
      precision = tp / (tp + fp),
      sensitivity = tp / (tp + fn), # Recall, TP rate
      specificity = tn / (tn + fp), # TN rate
      f1 = tp / (tp + 0.5 * (fp + fn))
      #f1 = 2 * precision * sensitivity / (precision + sensitivity)
    )

# Intended for training metrics (loss, network size, etc.).
# 'names' specifies a list of metric names to track.
class Metrics:
  def __init__(self, tensor, names, n_rec=0):
    self.names = names
    self.n_rec = n_rec # Current number of records
    self.storage = tensor # May be overallocated, hence n_rec

  def __getattr__(self, attr):
    try:
      idx = self.names.index(attr)
      return self.storage[:self.n_rec, idx]
    except ValueError:
      raise AttributeError(f"No attribute {attr}")

  def update(self, *args):
    for i in range(len(args)):
      self.storage[self.n_rec, i] = args[i]
    self.n_rec += 1


# Helper for recording/plotting performance averaged over all runs/folds;
# each dataset stores one of these for each model, after it is run.
@dataclass
class KFoldPerformance:
  confusions: Tensor # Sum over folds
  train_metrics: Metrics # Avg over folds # includes time and epoch
  n_folds: int
  n_runs: int
  test_metrics: PredictionMetrics = None

  def __post_init__(self):
    self.test_metrics = PredictionMetrics.from_confusion(self.confusions)

  def plot(self, title):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
    ax_loss, ax_score = axs

    fig.suptitle(title)
    ax_loss.set(title="Loss", ylabel="Loss")
    ax_loss.set_yscale('symlog', linthresh=1e-6)
    ax_score.set(title="Score", ylabel="Score")
    ax_loss.grid(which='both')
    ax_score.grid(which='both')

    ax_loss.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))

    trn = self.train_metrics
    tst = self.test_metrics

    # can change trn.epoch to trn.time
    plt.setp(axs, xlabel="Epoch") # xlabel="Time (s)"
    ax_loss.plot(trn.epoch, torch.stack([trn.train_loss, trn.test_loss], dim=1))
    ax_loss.legend(["Train", "Test"])

    ax_score.plot(trn.epoch, torch.stack([
      trn.train_accuracy,
      tst.accuracy,
      tst.precision,
      tst.sensitivity,
      tst.specificity,
      tst.f1
    ], dim=1))

    ax_score.legend([ 'Train Acc',
      'Acc', 'Prec',
      'Sens', 'Spec', 'F1'
    ], loc=(1.1, 0.3))

    for ax in [ax_loss, ax_score]:
      ax.relim()
      ax.autoscale_view()

    fig.tight_layout()
    return fig

# Handy for seeing what's going on as training progresses,
# without generating a separate plot for every fold/run.
class RealtimeTrainingFig:
  def __init__(self, xvar='epoch'):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
    ax_loss, ax_acc = axs

    fig.suptitle("Training Progress")
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False

    ax_loss.set(title="Loss", ylabel="Loss", ylim=(0, 1))
    ax_loss.set_yscale('symlog', linthresh=1e-6) # Symlog is linear near zero, log elsewhere

    ax_acc.set(title="Accuracy", ylabel="Accuracy", ylim=(0, 1.1))

    for ax in [ax_loss, ax_acc]:
      ax.set_xlabel(xvar)
      ax.grid(which='both')

    fig.tight_layout()

    self.fig = fig
    self.ax_loss = ax_loss
    self.ax_acc = ax_acc
    fig.canvas.draw()
    self.bg = fig.canvas.copy_from_bbox(fig.bbox)

    self.xvar = xvar
  
  def begin_fold(self, title):
    ax_loss, ax_acc = self.ax_loss, self.ax_acc
    # Set all previous lines to animated=False, so they're included in bg
    for line in ax_loss.lines+ax_acc.lines:
      line.set_animated(False)
    # Add training and test lines for this fold
    color=next(ax_loss._get_lines.prop_cycler)['color']
    line_args = {'animated': True, 'color': color, 'marker': '.', 'ms': 3}
    self.line_trainLoss, = ax_loss.plot([], [], **line_args)
    self.line_trainAcc, = ax_acc.plot([], [], **line_args)
    self.line_testLoss, = ax_loss.plot([], [], linestyle='--', **line_args)
    self.line_testAcc, = ax_acc.plot([], [], linestyle='--', **line_args)

    # Legend (simplest to overwrite each time) (and only need one)
    ax_acc.legend([self.line_trainAcc, self.line_testAcc], 
      ["Train", "Test"], loc='lower left')
    self.fig.suptitle(title)
    self.fig.canvas.draw()

    # Cache unchanging elements of figure for this fold.
    # Allows much faster updates ('blitting')
    self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

  def update(self, metrics):
    # Update plot
    xs = metrics.time if self.xvar == 'time' else metrics.epoch

    self.line_testAcc.set_data(xs, metrics.test_accuracy)
    self.line_testLoss.set_data(xs, metrics.test_loss)
    self.line_trainAcc.set_data(xs, metrics.train_accuracy)
    self.line_trainLoss.set_data(xs, metrics.train_loss)
    
    self.fig.canvas.restore_region(self.bg)
    self.ax_loss.draw_artist(self.line_trainLoss)
    self.ax_loss.draw_artist(self.line_testLoss)
    self.ax_acc.draw_artist(self.line_trainAcc)
    self.ax_acc.draw_artist(self.line_testAcc)

    self.fig.canvas.blit(self.fig.bbox)
    #fig.canvas.flush_events()
  
  def reset(self, xmax=None):
    [l.remove() for l in self.ax_loss.lines+self.ax_acc.lines]
    if xmax is not None:
      self.ax_loss.set_xlim(0, xmax)
      self.ax_acc.set_xlim(0, xmax)
