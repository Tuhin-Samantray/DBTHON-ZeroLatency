"""
Predictive Cross-Layer Transaction Placement - small prototype
Run:  pip install torch numpy pandas matplotlib
      python predictive_placement.py
Outputs: results.csv, forecast_plot.png, latency_plot.png, console tables
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v3" 
SEED = 42
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 1. SIMULATED DISTRIBUTED DB CLUSTER (monitoring layer data)
#    4 nodes x 4 metrics = 16 cross-layer features per timestep
#    metrics: network latency (ms), CPU util, queue length, replication lag (ms)
# ----------------------------------------------------------------------
N, T, METRICS = 4, 12000, ["lat", "cpu", "queue", "lag"]
F = N * len(METRICS)

def ar_noise(n, phi=0.9, sigma=1.0):
    e = rng.normal(0, sigma, n); x = np.zeros(n)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x

def generate_metrics():
    t = np.arange(T)
    base_lat = np.array([30, 35, 40, 45.0])
    data = np.zeros((T, N, 4))
    for n in range(N):
        load = 0.5 + 0.25 * np.sin(2 * np.pi * t / 1000 + n * 1.3)       # diurnal workload
        spike = np.zeros(T)                                               # congestion events
        for _ in range(T // 120):
            s = rng.integers(50, T - 120); ramp, hold = 15, rng.integers(15, 40)
            shape = np.concatenate([np.linspace(0, 1, ramp), np.ones(hold), np.linspace(1, 0, ramp)])
            spike[s:s + len(shape)] += shape * rng.uniform(0.6, 1.0)
        spike = np.clip(spike, 0, 1.2)
        cpu = np.clip(load + 0.3 * spike + 0.03 * ar_noise(T), 0.02, 0.99)
        # CROSS-LAYER LEAD-LAG: network latency reacts ~10 steps AFTER CPU/queue pressure starts
        spike_lat = np.roll(spike, 10); spike_lat[:10] = 0
        lat = base_lat[n] + 10 * load + 90 * spike_lat + 1.5 * ar_noise(T)
        queue = np.clip(5 + 25 * cpu ** 2 + 20 * spike + 1.0 * ar_noise(T), 0, None)
        lag = np.clip(10 + 30 * cpu + 15 * spike + 1.0 * ar_noise(T), 0, None)
        data[:, n] = np.stack([lat, cpu, queue, lag], axis=1)
    return data  # (T, N, 4)

raw = generate_metrics()
X_all = raw.reshape(T, F)

# chronological split 70 / 15 / 15
t1, t2 = int(0.70 * T), int(0.85 * T)
mu, sd = X_all[:t1].mean(0), X_all[:t1].std(0) + 1e-6
Xn = (X_all - mu) / sd

# ----------------------------------------------------------------------
# 2. PREDICTION ENGINE (GRU forecaster, horizon H steps ahead)
# ----------------------------------------------------------------------
W, H = 30, 10   # look-back window, prediction horizon

def make_windows(arr, lo, hi):
    idx = np.arange(lo, hi - W - H)
    X = np.stack([arr[i:i + W] for i in idx])
    y = np.stack([arr[i + W - 1 + H] for i in idx])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx

Xtr, ytr, _ = make_windows(Xn, 0, t1)
Xva, yva, _ = make_windows(Xn, t1, t2)
Xte, yte, idx_te = make_windows(Xn, t2, T)

class Forecaster(nn.Module):
    def __init__(self, f=F, h=64):
        super().__init__()
        self.gru = nn.GRU(f, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(h, f)
    def forward(self, x):
        out, _ = self.gru(x)
        return x[:, -1] + self.head(out[:, -1])   # predict change from last observation

model = Forecaster()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
EPOCHS, BS = 15, 128
best, best_state = 1e9, None
print("Training forecaster...")
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), BS):
        b = perm[i:i + BS]
        opt.zero_grad(); loss = loss_fn(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): v = loss_fn(model(Xva), yva).item()
    if v < best: best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    print(f"  epoch {ep + 1:02d}/{EPOCHS}  val MSE {v:.4f}")
model.load_state_dict(best_state); model.eval()
torch.save(model.state_dict(), f"{VERSION}_forecaster.pt")

with torch.no_grad(): pred_te = model(Xte).numpy()
pred_te = pred_te * sd + mu                      # back to real units
true_te = yte.numpy() * sd + mu
persist_te = Xte[:, -1].numpy() * sd + mu        # reactive baseline = "future equals now"

mae_model = np.abs(pred_te - true_te).reshape(-1, N, 4).mean((0, 1))
mae_pers = np.abs(persist_te - true_te).reshape(-1, N, 4).mean((0, 1))
fc = pd.DataFrame({"metric": METRICS, "MAE_persistence(reactive)": mae_pers, "MAE_GRU": mae_model})
fc["improvement_%"] = 100 * (1 - fc["MAE_GRU"] / fc["MAE_persistence(reactive)"])
print("\n=== Forecast accuracy (test set, horizon = %d steps) ===" % H)
print(fc.round(3).to_string(index=False))

# ----------------------------------------------------------------------
# 3. DECISION ENGINE: J = a*C_2PC + b*D_net + g*R_node
# ----------------------------------------------------------------------
ALPHA, BETA, GAMMA = 1.0, 2.0, 1.0
PENDING_PENALTY = 4.0     # ms per txn already routed to node in this step
REMOTE_PENALTY = 25.0     # ms per non-local partition (2PC overhead)

def cost_vector(state, remote_counts, pending):
    """state: (N,4) [lat,cpu,queue,lag]. Returns J for every node."""
    lat, cpu, queue, lag = state.T
    c2pc = REMOTE_PENALTY * remote_counts
    dnet = lat + 0.3 * lag
    rnode = queue + 80 * cpu ** 2 + PENDING_PENALTY * pending
    return ALPHA * c2pc + BETA * dnet + GAMMA * rnode

# ----------------------------------------------------------------------
# 4. EVALUATION: replay test period, route K txns per step with each strategy
# ----------------------------------------------------------------------
K = 20
strategies = ["random", "round_robin", "static_local", "reactive", "predictive_GRU", "oracle"]
lat_log = {s: [] for s in strategies}
share = {s: np.zeros(N) for s in strategies}
ev_rng = np.random.default_rng(7)

for j, i in enumerate(idx_te):
    t_now = i + W - 1                 # decision time
    t_exec = t_now + H                # execution time
    cur = raw[t_now]                  # observed state
    fut_true = raw[t_exec]            # state the txn really experiences
    fut_pred = pred_te[j].reshape(N, 4)
    # transactions: home partition + 30% chance of a 2nd partition
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
            elif s == "reactive": node = np.argmin(cost_vector(cur, remote, pending))
            elif s == "predictive_GRU": node = np.argmin(cost_vector(fut_pred, remote, pending))
            else: node = np.argmin(cost_vector(fut_true, remote, pending))
            # actual experienced latency uses the TRUE future state
            lat_log[s].append(cost_vector(fut_true, remote, pending)[node])
            pending[node] += 1; share[s][node] += 1

rows = []
for s in strategies:
    a = np.array(lat_log[s]); sh = share[s] / share[s].sum()
    rows.append({"strategy": s, "mean_ms": a.mean(), "p95_ms": np.percentile(a, 95),
                 "p99_ms": np.percentile(a, 99), "load_std(lower=balanced)": sh.std()})
res = pd.DataFrame(rows)
base = res.loc[res.strategy == "reactive", "mean_ms"].iloc[0]
res["vs_reactive_%"] = 100 * (1 - res["mean_ms"] / base)
print("\n=== Placement performance (lower latency is better) ===")
print(res.round(3).to_string(index=False))
res.to_csv(f"{VERSION}_results.csv", index=False)

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))
k = 0  # node 0 latency feature
n_show = 300
plt.plot(true_te[:n_show, k], label="True lat (node0)")
plt.plot(pred_te[:n_show, k], label="GRU forecast", alpha=.8)
plt.plot(persist_te[:n_show, k], label="Reactive (current value)", alpha=.6, ls="--")
plt.xlabel("test step"); plt.ylabel("ms"); plt.legend(); plt.title("Forecast vs reality, horizon=%d" % H)
plt.tight_layout(); plt.savefig(f"{VERSION}_forecast_plot.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(res.strategy, res.mean_ms); ax[0].set_title("Mean transaction cost (ms)"); ax[0].tick_params(axis="x", rotation=40)
ax[1].bar(res.strategy, res.p99_ms, color="tab:orange"); ax[1].set_title("p99 transaction cost (ms)"); ax[1].tick_params(axis="x", rotation=40)
plt.tight_layout(); plt.savefig(f"{VERSION}_latency_plot.png", dpi=150)
print(f"\nSaved: {VERSION}_results.csv, {VERSION}_forecast_plot.png, {VERSION}_latency_plot.png, {VERSION}_forecaster.pt")
