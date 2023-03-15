from pygam import te
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import lightgbm as lgb
from decorators import timer


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class MLPLinear2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear2, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.0):
        super(MLP, self).__init__()
        assert num_layers >= 2

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))

        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))

        self.linears.append(nn.Linear(hid_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

    def forward(self, x):
        for linear, bn in zip(self.linears[:-1], self.bns):
            x = linear(x)
            x = F.relu(x, inplace=True)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return F.log_softmax(x, dim=-1)


class LabelPropagation(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """

    def __init__(self, num_layers, alpha, adj="DAD"):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj

    @torch.no_grad()
    def forward(
        self,
        g,
        labels,
        mask=None,
        post_step=lambda y: y.clamp_(0.0, 1.0),
        num_layers=50,
        EI=1,
        confidence=1,
    ):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)

            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]

            last = (1 - self.alpha) * y * confidence
            degs = g.in_degrees().float().clamp(min=1)
            norm = (
                torch.pow(degs, -0.5 if self.adj == "DAD" else -1)
                .to(labels.device)
                .unsqueeze(1)
            )

            for _ in range(num_layers):
                # Assume the graphs to be undirected
                if self.adj in ["DAD", "AD"]:
                    y = norm * y

                g.ndata["h"] = y * EI
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                y = self.alpha * g.ndata.pop("h")

                if self.adj in ["DAD", "DA"]:
                    y = y * norm

                y = post_step(last + y)

            return y


class GNPD(nn.Module):
    r"""
    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        correction_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        smoothing_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """

    def __init__(
        self,
        num_correction_layers,
        correction_alpha,
        correction_adj,
        num_smoothing_layers,
        smoothing_alpha,
        smoothing_adj,
        autoscale=True,
        scale=1.0,
    ):
        super(GNPD, self).__init__()

        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(
            num_correction_layers, correction_alpha, correction_adj
        )
        self.prop2 = LabelPropagation(
            num_smoothing_layers, smoothing_alpha, smoothing_adj
        )
    @timer
    def correct(self, g, y_soft, y_true, mask, num_layers=50, y_soft2=None):

        with g.local_scope():
            if y_soft2 is None:
                y_soft2 = y_soft
            # assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)

            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]
            confidence = torch.max(y_soft, dim=1)[0]
            confidence = confidence / max(confidence)
            if self.autoscale:
                smoothed_error = self.prop1(
                    g,
                    error,
                    post_step=lambda x: x.clamp_(-1.0, 1.0),
                    num_layers=num_layers,
                )
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft2 + scale * smoothed_error * (1 - confidence).view(-1, 1)
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:

                def fix_input(x):
                    x[mask] = error[mask]
                    return x

                smoothed_error = self.prop1(
                    g, error, post_step=fix_input, num_layers=num_layers
                )

                result = y_soft + self.scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    @timer
    def smooth(self, g, y_soft, y_true, mask, use_labels=False, num_layers=50):

        with g.local_scope():
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)

            if use_labels:
                y_soft[mask] = y_true
            if use_labels:
                row, col = g.edges()
                i_lab = 0
                e_lab = 0
                for i in range(0, len(row), 200000):
                    summed_lab = (
                        y_soft[row[i : i + 200000], :] + y_soft[col[i : i + 200000], :]
                    )
                    i_lab += (summed_lab == 2).type(torch.int32).sum(dim=0)
                    e_lab += (summed_lab == 1).type(torch.int32).sum(dim=0)
                EI_lab = (e_lab - i_lab) / (e_lab + i_lab)

                EI_lab = 1 + (EI_lab) * 0.5

                EI_lab[EI_lab.isnan()] = 0
                EI = EI_lab.view(1, -1)
                confidence = 1
            else:
                confidence = torch.max(y_soft, dim=1)[0]
                confidence = (confidence / max(confidence)).view(-1, 1)
                confidence = 1
                EI = 1

            return self.prop2(
                g, y_soft, num_layers=num_layers, EI=EI, confidence=confidence
            )

    @timer
    def lightgbm(self, g, features, y_true, mask):
        clf = lgb.LGBMClassifier(
            num_leaves=50,
            learning_rate=0.01,
            n_estimators=300,
            feature_fraction=0.4,
            bagging_fraction=0.01,
        )
        clf.fit(features[mask].detach(), y_true.squeeze())
        return torch.from_numpy(clf.predict_proba(features.detach()))


    @timer
    def linear(self, g, text_features, features, y_true, mask, text_model):

        features = torch.cat([features, text_features], dim=1)
        aggregator = MLPLinear2(features.shape[1], text_features.shape[1])

        optimizer = torch.optim.Adam(aggregator.parameters(), lr=0.001)

        batch_size = 2048
        epochs = 60

        X_train = features[mask]
        for _ in range(1, epochs):
            permutation = torch.randperm(len(X_train))
            for i in range(0, X_train.size()[0], batch_size):
                text_model.train()
                optimizer.zero_grad()
                indices = permutation[i : i + batch_size]

                add = aggregator(X_train[indices].detach())
                added = text_features[indices] + add
                out = text_model(added)

                loss = F.nll_loss(out, y_true.squeeze(1)[indices])

                loss.backward()
                optimizer.step()

        text_model.cpu().eval()
        aggregator.cpu()
        return text_model(text_features + aggregator(features)).exp()
