import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- make dataset ---
X, y = make_moons(n_samples=2000, noise=0.25)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtr, ytr, Xte, yte = [t.to(device) for t in (Xtr, ytr, Xte, yte)]

# --- model rolled feedback with sigmoid gate ---
class RolledFeedbackNet(nn.Module):
    def __init__(self, input_dim=2, state_dim=128, n_blocks=3, feedback_scale=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(),
                nn.LayerNorm(state_dim)
            )
            for _ in range(n_blocks)
        ])
        self.readout = nn.Linear(state_dim, 2)      # logits
        self.fb_to_input = nn.Linear(state_dim, input_dim)
        self.gate_layer = nn.Linear(state_dim, input_dim)  # gate layer for feedback
        self.feedback_scale = feedback_scale
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        h = F.relu(self.input_proj(x))
        for b in self.blocks:
            delta = b(h)
            h = h + delta
        logits = self.readout(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_once(x)
        return logits

    def infer_loop(self, x0, y_true=None, max_iters=50, tol=1e-10, verbose=False, return_best=False):
        x = x0
        prev_logits = None
        logits_list = []
        best_acc = -1.0
        best_logits = None
        best_iter_idx = -1    # track index of best iteration

        for t in range(max_iters):
            logits, h = self.forward_once(x)
            logits_list.append(logits.detach().cpu().numpy())

            if return_best and (y_true is not None):
                preds = logits.argmax(dim=1)
                acc = (preds == y_true).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_logits = logits.detach()
                    best_iter_idx = t

            # gated feedback:
            gate = self.sigmoid(self.gate_layer(h))
            fb = gate * torch.tanh(self.fb_to_input(h)) * self.feedback_scale
            x = x + fb

            if prev_logits is not None:
                if torch.max(torch.abs(logits - prev_logits)) < tol:
                    if verbose:
                        print(f"Converged at iter {t}")
                    break
            prev_logits = logits

        if return_best and best_logits is not None:
            return best_logits, logits_list, best_iter_idx
        else:
            return logits, logits_list, None


loss_fn = nn.CrossEntropyLoss()

def eval_acc(model):
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        acc = (logits.argmax(dim=1) == yte).float().mean().item()
    return acc

def eval_acc_iterative(model, batch_size=128, max_iters=50, tol=1e-5):
    model.eval()
    total_samples = 0
    correct_per_iter = None  
    with torch.no_grad():
        for i in range(0, Xte.size(0), batch_size):
            xb = Xte[i:i+batch_size]
            yb = yte[i:i+batch_size]
            _, logits_list, _ = model.infer_loop(xb, max_iters=max_iters, tol=tol, return_best=False)
            n_steps = len(logits_list)
            if correct_per_iter is None:
                correct_per_iter = np.zeros(n_steps)
            for step_idx, logits_np in enumerate(logits_list):
                preds = np.argmax(logits_np, axis=1)
                correct_per_iter[step_idx] += np.sum(preds == yb.cpu().numpy())
            total_samples += yb.size(0)
    acc_per_iter = (correct_per_iter / total_samples).tolist()
    return acc_per_iter

def eval_acc_iterative_best(model, batch_size=128, max_iters=50, tol=1e-5):
    model.eval()
    total_samples = 0
    total_correct = 0
    best_iters_all = []  # store best iter per batch
    batch_sizes = []

    with torch.no_grad():
        for i in range(0, Xte.size(0), batch_size):
            xb = Xte[i:i+batch_size]
            yb = yte[i:i+batch_size]

            best_logits, _, best_iter = model.infer_loop(
                xb, y_true=yb, max_iters=max_iters, tol=tol, return_best=True
            )
            preds = best_logits.argmax(dim=1)
            correct = (preds == yb).sum().item()
            total_samples += yb.size(0)
            total_correct += correct
            best_iters_all.append(best_iter)
            batch_sizes.append(yb.size(0))

    overall_acc = total_correct / total_samples
    best_iter_global = int(np.round(np.average(best_iters_all, weights=batch_sizes)))
    return overall_acc, best_iter_global

def train_epoch_unrolled(model, opt, batch_size=128, max_iters=10, tol=1e-10):
    model.train()
    perm = torch.randperm(Xtr.size(0))
    total_loss = 0.0
    for i in range(0, Xtr.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb = Xtr[idx]
        yb = ytr[idx]

        opt.zero_grad()
        x = xb
        losses = []
        prev_logits = None

        for t in range(max_iters):
            logits, h = model.forward_once(x)
            loss = loss_fn(logits, yb)
            losses.append(loss)

            # gated feedback during training
            gate = model.sigmoid(model.gate_layer(h))
            fb = gate * torch.tanh(model.fb_to_input(h)) * model.feedback_scale
            x = x + fb

            if prev_logits is not None:
                if torch.max(torch.abs(logits - prev_logits)) < tol:
                    break
            prev_logits = logits

        total_loss_batch = torch.stack(losses).mean()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(total_loss_batch.item()) * xb.size(0)
    return total_loss / Xtr.size(0)

# Train model
model = RolledFeedbackNet(feedback_scale=0.3).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
epochs = 60

for ep in range(1, epochs+1):
    loss = train_epoch_unrolled(model, opt)
    if ep % 10 == 0 or ep == 1:
        print(f"Epoch {ep}/{epochs} | Loss: {loss:.4f}")

print("Training complete.\n")

# Accuracy
single_acc = eval_acc(model)
iter_best_acc, best_iter = eval_acc_iterative_best(model)
print(f"Final Single-step Test Accuracy: {single_acc:.4f}")
print(f"Final Iterative Best Test Accuracy: {iter_best_acc:.4f} at iteration {best_iter}")

# Plot decision boundaries
def plot_decision_boundary(model, X, y, predict_fn, title, subplot_idx):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32).to(device)

    preds = predict_fn(grid_t)
    Z = preds.cpu().numpy().reshape(xx.shape)

    plt.subplot(1, 3, subplot_idx)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.7)
    plt.scatter(X[:,0], X[:,1], c=y, s=10, cmap=plt.cm.Set1, edgecolors='k')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

def predict_single_step(x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return logits.argmax(dim=1)

def predict_iterative_last(x, max_iters=50):
    model.eval()
    with torch.no_grad():
        logits, _, _ = model.infer_loop(x, max_iters=max_iters, return_best=False)
        return logits.argmax(dim=1)

def predict_iterative_best(x, max_iters=50):
    model.eval()
    with torch.no_grad():
        best_logits, _, _ = model.infer_loop(x, max_iters=max_iters, return_best=True)
        return best_logits.argmax(dim=1)

plt.figure(figsize=(18,5))
Xnp = Xte.cpu().numpy()
ynp = yte.cpu().numpy()

plot_decision_boundary(model, Xnp, ynp, predict_single_step, "Single-step Decision Boundary", 1)
plot_decision_boundary(model, Xnp, ynp, predict_iterative_last, "Iterative (Last Step) Decision Boundary", 2)
plot_decision_boundary(model, Xnp, ynp, predict_iterative_best, "Iterative (Best Step) Decision Boundary", 3)

plt.tight_layout()
plt.show()
