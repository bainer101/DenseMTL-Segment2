import torch
import torch.nn.functional as F
from loss.setup_loss import SetupLoss
from utils.train import resize_input
from utils.logs import cpprint
import metrics


class FluidAwareLayerLoss(SetupLoss):
    def __init__(self, scales, n_classes, weight=1):
        meters = {f'layer_seg_{s}': metrics.Semseg(n_classes) for s in scales}
        super().__init__(scales, weight, **meters)

        self.n_classes = n_classes
        self.maps = meters.keys()
        self.eval_meter = meters[f'layer_seg_{scales[-1]}']

        cpprint(f'FluidAwareLayerLoss init'
                f'\n Scales: {list((s, w.item()) for s, w in zip(self.scales, self.weight))}'
                f'\n Evaluating on scale: {scales[-1]}', c='yellow')

    def main_metric(self):
        return 'mIoU', self.eval_meter.mIoU, 0

    def loss_fn(self, pred, target, fluid_mask):
        ce = F.cross_entropy(pred, target, reduction='none', ignore_index=250)
        masked_loss = ce * (1.0 - fluid_mask.squeeze(1))  # Suppress loss in fluid
        return masked_loss.mean()

    def evaluate(self, pred, target, scale):
        self.metrics[f'layer_seg_{scale}'].measure(label_trues=target, label_preds=pred)

    def forward(self, x, y, viz, train=True):
        loss = self._init_losses('total')
        layer_gt = x['layer_seg']
        fluid_mask = x['fluid_seg']

        maps = dict(layer_seg_gt=viz.seg(layer_gt))

        for s in self.scales:
            pred = y['layer_seg', s]
            pred_up = resize_input(pred, layer_gt, align_corners=True)
            maps[f'layer_seg_{s}'] = viz.seg(pred_up.argmax(1))

            loss[f'layer_seg_{s}'] = self.loss_fn(pred_up, layer_gt, fluid_mask)

            if not train:
                self.evaluate(pred_up.argmax(1), layer_gt, s)

        loss.total = self.mean([loss[k] for k in self.maps])
        return loss, maps
