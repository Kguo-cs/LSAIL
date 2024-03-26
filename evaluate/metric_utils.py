from copy import deepcopy
import torchmetrics
from pytorch_lightning.utilities.apply_func import move_data_to_device

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
import numpy as np

import heapq
from functools import partial


val_metrics = {
  'pos_diff_mean': torchmetrics.MeanMetric(),
  'x_diff_mean': torchmetrics.MeanMetric(),
  'y_diff_mean': torchmetrics.MeanMetric(),
  'yaw_diff_mean': torchmetrics.MeanMetric(),

  "road_density_rmse": torchmetrics.MeanMetric(),
  "road_speed_rmse": torchmetrics.MeanMetric(),

  'off_road': torchmetrics.MeanMetric(),
  "off_road_longterm":torchmetrics.MeanMetric(),
  "discomfort_longterm": torchmetrics.MeanMetric(),
  "discomfort": torchmetrics.MeanMetric(),

  'min_ade': torchmetrics.MeanMetric(),
  'speed_rmse': torchmetrics.MeanMetric(),
  'pos_rmse': torchmetrics.MeanMetric(),

  'acc_mean': torchmetrics.MeanMetric(),
  'jerk_mean': torchmetrics.MeanMetric(),
  'comfort': torchmetrics.MeanMetric(),
}


train_metrics = {
  'loss': torchmetrics.MeanMetric(),
  'bc_loss':torchmetrics.MeanMetric(),
  'expert_rec_loss': torchmetrics.MeanMetric(),
  'expert_kl_loss': torchmetrics.MeanMetric(),
  "learner_rec_loss":  torchmetrics.MeanMetric(),
  'learner_kl_loss': torchmetrics.MeanMetric(),
}


class LazyEval:
  def __init__(self, func, *args, **kwargs):
    self.func = func
    self.args = args
    self.kwargs = kwargs

  def __repr__(self):
    return F"Func: {{{self.func}}}; args: {{{self.args}}}; kwargs: {{{self.kwargs}}}"

  @staticmethod
  def evaluate(lazy_obj):
    assert isinstance(lazy_obj, LazyEval), F"Can only evaluate LazyEval, but got {lazy_obj}"

    # nested lazy eval
    if isinstance(lazy_obj.func, LazyEval):
      lazy_obj.func = LazyEval.evaluate(lazy_obj.func)
    lazy_obj.args = [LazyEval.evaluate(arg) if isinstance(
        arg, LazyEval) else arg for arg in lazy_obj.args]
    lazy_obj.kwargs = {k: (LazyEval.evaluate(v) if isinstance(v, LazyEval) else v)
                       for k, v in lazy_obj.kwargs.items()}

    return lazy_obj.func(*lazy_obj.args, **lazy_obj.kwargs)

  def __call__(self):
    return self

  def __getattr__(self, attr):
    return partial(LazyEval, LazyEval(getattr, self, attr))


def get_device(val):
  if isinstance(val, torch.Tensor):
    return val.device
  elif isinstance(val, dict):
    for v in val.values():
      res = get_device(v)
      if res is not None:
        return res
  elif '__iter__' in dir(val) or isinstance(val, list) or isinstance(val, tuple):
    for v in val:
      res = get_device(v)
      if res is not None:
        return res

  return None


class DictMetric(torchmetrics.Metric):

  def __init__(self, metric_dict={}, prefix='', allow_auto_add=True, default_metrics=torchmetrics.MeanMetric):
    super().__init__()

    self.metric_dict = nn.ModuleDict(metric_dict)
    self._prefix = prefix
    self.allow_auto_add = allow_auto_add
    self.default_metrics = default_metrics

  @property
  def prefix(self):
    return self._prefix

  @prefix.setter
  def prefix(self, new_prefix):
    assert isinstance(new_prefix, str), F"Prefix needs to be str, instead got {new_prefix}"
    self._prefix = new_prefix

  def forward(self, update_metric_dict):
    if not self.allow_auto_add:
      # partial forward
      assert len(set(update_metric_dict.keys()) - set(self.metric_dict.keys())) == 0,\
          F"{set(update_metric_dict.keys())} not in {set(self.metric_dict.keys())}"

    res = {}
    for k, v in update_metric_dict.items():
      if self.allow_auto_add and k not in self.metric_dict:
        device = get_device(v)
        self.metric_dict[k] = self.default_metrics().to(device)
      res[self.prefix + k] = self.metric_dict[k](v)

    return res

  def update(self, update_metric_dict):
    if not self.allow_auto_add:
      # partial forward
      assert len(set(update_metric_dict.keys()) - set(self.metric_dict.keys())) == 0,\
          F"{set(update_metric_dict.keys())} not in {set(self.metric_dict.keys())}"

    for k, v in update_metric_dict.items():
      if self.allow_auto_add and k not in self.metric_dict:
        device = get_device(v)
        self.metric_dict[k] = self.default_metrics().to(device)
      self.metric_dict[k].update(v)

  def compute(self, keys=None):
    # partial compute
    if keys is None:
      res = {self.prefix + k: safe_compute(v) for k, v in self.metric_dict.items()}
    else:
      res = {self.prefix + k: safe_compute(self.metric_dict[k]) for k in keys}
    res = {k: v for k, v in res.items() if v is not None}
    return res

  def reset(self, keys=None):
    # partial compute
    if keys is None:
      return {self.prefix + k: v.reset() for k, v in self.metric_dict.items()}
    else:
      return {self.prefix + k: self.metric_dict[k].reset() for k in keys}

  def clone(self, prefix=''):
    metrics = deepcopy(self)
    metrics.prefix = prefix

    return metrics


def safe_compute(metric, default_val=None):
  try:
    return metric.compute()
  except Exception as e:
    return e

def gaussian_2d(norm: torch.Tensor, s: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:

    norm1 = norm[..., 0]
    norm2 = norm[..., 1]

    s1 = s[..., 0]
    s2 = s[..., 1]

    s1s2 = s1 * s2

    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2

    neg_rho = 1 - rho ** 2

    # log_prob=-z/(2*neg_rho)-torch.log(s1s2)-1/2*torch.log(neg_rho)

    # ent= 1/2*torch.log(neg_rho)+torch.log(s1s2)#+(1+torch.log(2*np.pi))
    ent = 1 / 2 * torch.log(neg_rho) + torch.log(s1s2) + np.log(2 * np.pi)

    neg_log_prob_ent = z / (2 * neg_rho)  # neg_(log_prob+ent)

    neg_log_prob = neg_log_prob_ent + ent

    return neg_log_prob


