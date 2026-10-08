"""
v4: Graph-aware forecaster (simplified STGCN-style) vs plain GRU vs reactive.
New in v4:
  * Congestion SPILLS OVER from node i-1 to node i after ~8 steps (network link coupling)
  * GraphForecaster mixes each node's features with its ring neighbours' before the GRU
Run: python predictive_placement_v4.py
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v4"
SEED = 42
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 1. SIMULATED CLUSTER  (4 nodes x [lat, cpu, queue, lag])
# ----------------------------------------------------------------------
N, T, METRICS = 4, 12000, ["lat", "cpu", "queue", "lag"]
F = N * len(METRICS)
SPILL_DELAY, SPILL_GAIN = 8, 0.6

def ar_noise(n, phi=0.9, sigma=1.0):
    e = rng.normal(0, sigma, n); x = np.zeros(n)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x

def make_spikes():
    spikes = np.zeros((N, T))
    for n in range(N):
        for _ in range(T // 120):
            s = rng.integers(50, T - 120); ramp, hold = 15, rng.integers(15, 40)
            shape = np.concatenate([np.linspace(0, 1, ramp), np.ones(hold), np.linspace(1, 0, ramp)])
            spikes[n, s:s + len(shape)] += shape * rng.uniform(0.6, 1.0)
    return np.clip(spikes, 0, 1.2)

def generate_metrics():
    t = np.arange(T)
    base_lat = np.array([30, 35, 40, 45.0])
    own = make_spikes()
    data = np.zeros((T, N, 4))
    for n in range(N):
        load = 0.5 + 0.25 * np.sin(2 * np.pi * t / 1000 + n * 1.3)
        # congestion at the upstream neighbour spills over to this node after a delay
        spill = SPILL_GAIN * np.roll(own[(n - 1) % N], SPILL_DELAY); spill[:SPILL_DELAY] = 0
        spike = np.clip(own[n] + spill, 0, 1.2)
        cpu = np.clip(load + 0.3 * spike + 0.03 * ar_noise(T), 0.02, 0.99)
        spike_lat = np.roll(spike, 10); spike_lat[:10] = 0    # latency reacts ~10 steps after CPU/queue
        lat = base_lat[n] + 10 * load + 90 * spike_lat + 1.5 * ar_noise(T)
        queue = np.clip(5 + 25 * cpu ** 2 + 20 * spike + 1.0 * ar_noise(T), 0, None)
        lag = np.clip(10 + 30 * cpu + 15 * spike + 1.0 * ar_noise(T), 0, None)
        data[:, n] = np.stack([lat, cpu, queue, lag], axis=1)
    return data

raw = generate_metrics()
X_all = raw.reshape(T, F)
t1, t2 = int(0.70 * T), int(0.85 * T)
mu, sd = X_all[:t1].mean(0), X_all[:t1].std(0) + 1e-6
Xn = (X_all - mu) / sd

# ----------------------------------------------------------------------
# 2. PREDICTION ENGINE: plain GRU and graph-aware GRU
# ----------------------------------------------------------------------
W, H = 30, 10

def make_windows(arr, lo, hi):
    idx = np.arange(lo, hi - W - H)
    X = np.stack([arr[i:i + W] for i in idx])
    y = np.stack([arr[i + W - 1 + H] for i in idx])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx

Xtr, ytr, _ = make_windows(Xn, 0, t1)
Xva, yva, _ = make_windows(Xn, t1, t2)
Xte, yte, idx_te = make_windows(Xn, t2, T)

class Forecaster(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = nn.GRU(F, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(h, F)
    def forward(self, x):
        out, _ = self.gru(x)
        return x[:, -1] + self.head(out[:, -1])

class GraphForecaster(nn.Module):
    """Graph mixing over a ring of nodes (self + 2 neighbours), then GRU."""
    def __init__(self, h=64):
        super().__init__()
        A = np.eye(N) + np.roll(np.eye(N), 1, 0) + np.roll(np.eye(N), -1, 0)
        A = A / A.sum(1, keepdims=True)
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.gru = nn.GRU(F * 2, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(h, F)
    def forward(self, x):
        B, Wn, _ = x.shape
        x4 = x.reshape(B, Wn, N, 4)
        nb = torch.einsum("ij,bwjf->bwif", self.A, x4)       # neighbour-aggregated features
        z = torch.cat([x4, nb], dim=-1).reshape(B, Wn, N * 8)
        out, _ = self.gru(z)
        return x[:, -1] + self.head(out[:, -1])

EPOCHS, BS = 15, 128
def train(model, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss(); best, best_state = 1e9, None
    print(f"Training {name}...")
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            opt.zero_grad(); loss = loss_fn(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(Xva), yva).item()
        if v < best: best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"  [{name}] epoch {ep + 1:02d}/{EPOCHS}  val MSE {v:.4f}")
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), f"{VERSION}_{name}.pt")
    return model

def predict(m):
    with torch.no_grad(): return m(Xte).numpy() * sd + mu

m_gru = train(Forecaster(), "GRU")
m_gnn = train(GraphForecaster(), "GraphGRU")
preds = {"GRU": predict(m_gru), "GraphGRU": predict(m_gnn)}
true_te = yte.numpy() * sd + mu
persist_te = Xte[:, -1].numpy() * sd + mu

def mae(p): return np.abs(p - true_te).reshape(-1, N, 4).mean((0, 1))
fc = pd.DataFrame({"metric": METRICS, "MAE_persistence": mae(persist_te),
                   "MAE_GRU": mae(preds["GRU"]), "MAE_GraphGRU": mae(preds["GraphGRU"])})
fc["GRU_impr_%"] = 100 * (1 - fc.MAE_GRU / fc.MAE_persistence)
fc["GraphGRU_impr_%"] = 100 * (1 - fc.MAE_GraphGRU / fc.MAE_persistence)
print("\n=== Forecast accuracy (test set, horizon = %d steps) ===" % H)
print(fc.round(3).to_string(index=False))
fc.to_csv(f"{VERSION}_forecast_accuracy.csv", index=False)

# ----------------------------------------------------------------------
# 3. DECISION ENGINE: J = a*C_2PC + b*D_net + g*R_node
# ----------------------------------------------------------------------
ALPHA, BETA, GAMMA = 1.0, 2.0, 1.0
PENDING_PENALTY, REMOTE_PENALTY = 4.0, 25.0

def cost_vector(state, remote_counts, pending):
    lat, cpu, queue, lag = state.T
    return (ALPHA * REMOTE_PENALTY * remote_counts + BETA * (lat + 0.3 * lag)
            + GAMMA * (queue + 80 * cpu ** 2 + PENDING_PENALTY * pending))

# ----------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------
K = 20
strategies = ["random", "round_robin", "static_local", "reactive",
              "predictive_GRU", "predictive_GraphGRU", "oracle"]
lat_log = {s: [] for s in strategies}
share = {s: np.zeros(N) for s in strategies}
ev_rng = np.random.default_rng(7)

for j, i in enumerate(idx_te):
    t_now = i + W - 1; t_exec = t_now + H
    fut_true = raw[t_exec]
    state_for = {"reactive": raw[t_now],
                 "predictive_GRU": preds["GRU"][j].reshape(N, 4),
                 "predictive_GraphGRU": preds["GraphGRU"][j].reshape(N, 4),
                 "oracle": fut_true}
    homes = ev_rng.integers(0, N, K)
    seconds = np.where(ev_rng.random(K) < 0.3, ev_rng.integers(0, N, K), -1)
    for s in strategies:
        pending = np.zeros(N)
        for k in range(K):
            parts = {homes[k]} | ({seconds[k]} if seconds[k] >= 0 else set())
            remote = np.array([len(parts - {n}) for n in range(N)], dtype=float)
            if s == "random": node = ev_rng.integers(N)
            elif s == "round_robin": node = (j * K + k) % N
            elif s == "static_local": node = homes[k]
            else: node = np.argmin(cost_vector(state_for[s], remote, pending))
            lat_log[s].append(cost_vector(fut_true, remote, pending)[node])
            pending[node] += 1; share[s][node] += 1

rows = []
for s in strategies:
    a = np.array(lat_log[s]); sh = share[s] / share[s].sum()
    rows.append({"strategy": s, "mean_ms": a.mean(), "p95_ms": np.percentile(a, 95),
                 "p99_ms": np.percentile(a, 99), "load_std(lower=balanced)": sh.std()})
res = pd.DataFrame(rows)
rb = res.loc[res.strategy == "reactive", ["mean_ms", "p95_ms"]].iloc[0]
res["mean_vs_reactive_%"] = 100 * (1 - res["mean_ms"] / rb.mean_ms)
res["p95_vs_reactive_%"] = 100 * (1 - res["p95_ms"] / rb.p95_ms)
print("\n=== Placement performance (lower is better) ===")
print(res.round(3).to_string(index=False))
res.to_csv(f"{VERSION}_results.csv", index=False)

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4)); n_show = 300
plt.plot(true_te[:n_show, 0], label="True lat (node0)")
plt.plot(preds["GRU"][:n_show, 0], label="GRU", alpha=.8)
plt.plot(preds["GraphGRU"][:n_show, 0], label="GraphGRU", alpha=.8)
plt.plot(persist_te[:n_show, 0], label="Reactive (current value)", alpha=.5, ls="--")
plt.xlabel("test step"); plt.ylabel("ms"); plt.legend(); plt.title(f"Forecast vs reality, horizon={H}")
plt.tight_layout(); plt.savefig(f"{VERSION}_forecast_plot.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].bar(res.strategy, res.mean_ms); ax[0].set_title("Mean transaction cost (ms)"); ax[0].tick_params(axis="x", rotation=45)
ax[1].bar(res.strategy, res.p95_ms, color="tab:orange"); ax[1].set_title("p95 transaction cost (ms)"); ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(f"{VERSION}_latency_plot.png", dpi=150)
print(f"\nSaved: {VERSION}_results.csv, {VERSION}_forecast_accuracy.csv, {VERSION}_forecast_plot.png, {VERSION}_latency_plot.png, {VERSION}_GRU.pt, {VERSION}_GraphGRU.pt")
