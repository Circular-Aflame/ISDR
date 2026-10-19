from __future__ import annotations

from math import log2

import numpy as np
import torch

from .common import pca_reduce, project_torch_model, score_bins, set_seed, transform_pca
from .registry import register


class _NeuralMapper(torch.nn.Module):
    def __init__(self, dim_input: int, dim_emb: int = 2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_input),
            torch.nn.BatchNorm1d(dim_input),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_input, dim_input),
            torch.nn.BatchNorm1d(dim_input),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_input, dim_input),
            torch.nn.BatchNorm1d(dim_input),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_input, dim_emb),
        )

    def forward(self, x):
        return self.net(x)


def _pairwise_sq(x):
    sq = (x * x).sum(dim=1)
    return torch.clamp(sq[:, None] + sq[None, :] - 2.0 * x @ x.t(), min=0.0)


def _binary_search_p(distances, target_entropy, mask, max_iter=50):
    n = distances.shape[0]
    lo = torch.full((n,), 1e-8, device=distances.device)
    hi = torch.full((n,), 1e4, device=distances.device)
    sigma_sq = (lo + hi) / 2.0
    probs = torch.zeros_like(distances)
    for _ in range(max_iter):
        logits = -distances / (2.0 * sigma_sq[:, None].clamp_min(1e-8))
        probs = torch.exp(logits) * mask
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-10)
        entropy = -(probs.clamp_min(1e-10) * torch.log2(probs.clamp_min(1e-10))).sum(dim=1)
        too_high = entropy > target_entropy
        hi = torch.where(too_high, sigma_sq, hi)
        lo = torch.where(too_high, lo, sigma_sq)
        sigma_sq = (lo + hi) / 2.0
    return ((probs + probs.t()) / (2.0 * n)).clamp_min(1e-10)


def _joint_probabilities_batch(x, perplexity=None):
    n = x.shape[0]
    distances = _pairwise_sq(x)
    mask = 1.0 - torch.eye(n, device=x.device)
    if perplexity is None:
        max_entropy = int(round(log2(max(n / 2, 2))))
        p_joint = torch.zeros((n, n), device=x.device)
        count = 0
        for entropy in range(1, max(max_entropy, 2)):
            p_joint = p_joint + _binary_search_p(distances, float(entropy), mask)
            count += 1
        return p_joint / max(1, count)
    return _binary_search_p(distances, log2(float(perplexity)), mask)


def _q_joint(z):
    n = z.shape[0]
    mask = 1.0 - torch.eye(n, device=z.device)
    q = (1.0 + _pairwise_sq(z)).pow(-1.0) * mask
    return (q / q.sum().clamp_min(1e-10)).clamp_min(1e-10)


def _maploss(z, scores):
    n = z.shape[0]
    n_train = max(2, min(n - 1, int(0.9 * n)))
    perm = torch.randperm(n, device=z.device)
    train = perm[:n_train]
    val = perm[n_train:]
    z_tr = z[train]
    s_tr = scores[train]
    k_tr = 1.0 / (1.0 + torch.cdist(z_tr, z_tr).pow(2))
    pred_tr = (k_tr @ s_tr) / k_tr.sum(dim=1).clamp_min(1e-10)
    loss_tr = (s_tr - pred_tr).pow(2).mean()
    if val.numel() == 0:
        return loss_tr
    k_val = 1.0 / (1.0 + torch.cdist(z[val], z_tr).pow(2))
    pred_val = (k_val @ s_tr) / k_val.sum(dim=1).clamp_min(1e-10)
    loss_val = (scores[val] - pred_val).pow(2).mean()
    return 0.3 * loss_tr + loss_val


@register("akrmap")
def akrmap_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    set_seed(seed)
    params = params or {}
    alpha = float(np.clip(float(params.get("alpha", 1.0 / 9.0)), 0.0, 1.0))
    X_pca, pca = pca_reduce(X_train, int(params.get("pca_dim", 100)), seed)
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    model = _NeuralMapper(X_pca.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 0.002)))
    x = torch.as_tensor(X_pca, dtype=torch.float32, device=device)
    scores = torch.as_tensor(scores_train, dtype=torch.float32, device=device).reshape(-1)
    dataset = torch.utils.data.TensorDataset(x, scores)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(params.get("batch_size", 1000)), shuffle=True, drop_last=False
    )
    for _ in range(int(params.get("n_epochs", 20))):
        for xb, sb in loader:
            if xb.shape[0] < 4:
                continue
            z = model(xb)
            p_joint = _joint_probabilities_batch(xb, params.get("perplexity", None))
            loss_kl = (p_joint * torch.log(p_joint / _q_joint(z))).sum()
            loss_map = _maploss(z, sb) if alpha > 0 else torch.tensor(0.0, device=device)
            loss = (1.0 - alpha) * loss_kl + alpha * loss_map
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    Z_train = project_torch_model(model, X_pca, device)
    Z_test = project_torch_model(model, transform_pca(pca, X_test), device) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}


@register("rfae")
@register("rf_ae")
@register("rf-ae")
def rfae_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    try:
        from rfae import RFAE
    except ImportError as exc:
        raise ImportError("RF-AE requires the public `rfae` package.") from exc

    set_seed(seed)
    params = params or {}
    if int(params.get("pca_dim", 0) or 0) > 0:
        raise ValueError("RF-AE uses raw standardized input; set n_pca for its RF-PHATE embedder.")
    X_fit = np.asarray(X_train, dtype=np.float32)
    scores = np.asarray(scores_train).reshape(-1)
    prediction_type = params.get("prediction_type")
    if prediction_type is None and np.unique(scores).size <= 20 and np.allclose(scores, np.round(scores)):
        prediction_type = "classification"
        y_fit = scores.astype(int)
    else:
        prediction_type = str(prediction_type or "regression")
        y_fit = scores.astype(int) if prediction_type == "classification" else scores
    rfphate_n_pca = max(1, int(params.get("rfphate_n_pca", params.get("n_pca", 100))))
    n_landmark = max(2, min(int(params.get("n_landmark", 2000)), X_fit.shape[0] - 1))
    embedder_params = {
        "prediction_type": prediction_type,
        "n_pca": rfphate_n_pca,
        "n_landmark": n_landmark,
        "n_estimators": int(params.get("n_estimators", 500)),
        "random_state": seed,
        "n_jobs": int(params.get("n_jobs", -1)),
        "verbose": 0,
    }
    embedder_params.update(params.get("embedder_params", {}))
    model = RFAE(
        n_components=2,
        batch_size=int(params.get("batch_size", 256)),
        lr=float(params.get("lr", 1e-3)),
        random_state=seed,
        device=None if device == "auto" else device,
        epochs=int(params.get("epochs", params.get("n_epochs", 50))),
        lam=float(params.get("lam", 0.01)),
        embedder_params=embedder_params,
    )
    Z_train = model.fit_transform(X_fit, y_fit).astype(np.float32)
    Z_test = model.transform(np.asarray(X_test, dtype=np.float32)).astype(np.float32) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}


class _TPSEMLP(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_classes: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        h1 = max(64, min(hidden_dim, input_dim))
        h2 = max(32, h1 // 2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h2, latent_dim),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, max(16, h2 // 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(16, h2 // 2), n_classes),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.classifier(z), self.decoder(z)


@register("tpse")
@register("tp-se")
def tpse_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    set_seed(seed)
    params = params or {}
    X_pca, pca = pca_reduce(X_train, int(params.get("pca_dim", 128)), seed)
    labels = score_bins(scores_train, int(params.get("n_classes", 10)))
    n_classes = int(labels.max() + 1)
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    model = _TPSEMLP(
        X_pca.shape[1], 2, n_classes, int(params.get("hidden_dim", 256)), float(params.get("dropout", 0.2))
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(params.get("lr", 1e-3)), weight_decay=float(params.get("weight_decay", 1e-5))
    )
    x = torch.as_tensor(X_pca, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=int(params.get("batch_size", 512)), shuffle=True
    )
    alpha = float(np.clip(float(params.get("alpha", 0.5)), 0.0, 1.0))
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    for _ in range(int(params.get("n_epochs", 40))):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, logits, recon = model(xb)
            opt.zero_grad(set_to_none=True)
            loss = (1.0 - alpha) * mse(recon, xb) + alpha * ce(logits, yb)
            loss.backward()
            opt.step()
    Z_train = project_torch_model(model.encoder, X_pca, device)
    Z_test = project_torch_model(model.encoder, transform_pca(pca, X_test), device) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}
