import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Norton data
# ------------------------------------------------------------------
NORTON = {
    "1A": {"inner": 1.00000, "radius": 1.0},
    "2A": {"inner": 0.25000, "radius": 5.5},
    "2B": {"inner": 0.00000, "radius": 8.2},
    "3A": {"inner": 0.05078, "radius": 10.6},
    "3C": {"inner": 0.01562, "radius": 12.0},
    "4A": {"inner": 0.03125, "radius": 15.0},
    "4B": {"inner": 0.00781, "radius": 14.0},
    "5A": {"inner": 0.02344, "radius": 16.0},
    "6A": {"inner": 0.01953, "radius": 17.0},
}
CLASSES = list(NORTON.keys())
N_CLASSES = len(CLASSES)

# ------------------------------------------------------------------
# Graded residual
# ------------------------------------------------------------------
class GradedResidual(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.grades = [1, 16, 32, dim - 49]
        self.tau_net = nn.Sequential(nn.Linear(dim + N_CLASSES, 64), nn.Tanh(), nn.Linear(64, 1))
        self.f_net = nn.Sequential(nn.Linear(dim + N_CLASSES, 128), nn.Tanh(), nn.Linear(128, dim))
        self.inner_head = nn.Linear(dim + N_CLASSES, 1)

    def barrier_force(self, x, class_id, strength):
        r = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        target = torch.tensor([NORTON[CLASSES[i]]["radius"] for i in class_id],
                               device=x.device, dtype=x.dtype).unsqueeze(-1)
        return -strength * (r - target) * (x / r)

    def step(self, x, onehot, class_id, interaction, strength):
        tau = F.softplus(self.tau_net(torch.cat([x, onehot], -1))) + 0.1
        f = self.f_net(torch.cat([x, onehot], -1))
        barrier = self.barrier_force(x, class_id, strength)
        return x + 0.1 * ((-x / tau) + f + barrier + interaction)

    def predict_inner(self, x, onehot):
        return self.inner_head(torch.cat([x, onehot], -1)).squeeze(-1)

# ------------------------------------------------------------------
# Sparse graph
# ------------------------------------------------------------------
class GraphCoupling(nn.Module):
    def __init__(self, n_axes=8):
        super().__init__()
        self.n_axes = n_axes
        self.edge_logits = nn.Parameter(torch.zeros(n_axes, n_axes))
        mask = torch.zeros(n_axes, n_axes)
        for i in range(n_axes):
            mask[i, (i+1) % n_axes] = 1
            mask[i, (i-1) % n_axes] = 1
            mask[i, (i+2) % n_axes] = 1
        self.register_buffer("mask", mask)

    def forward(self, states):
        w = torch.sigmoid(self.edge_logits) * self.mask
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
        msg = torch.einsum("ij,bjd->bid", w, states)
        return 0.15 * (msg - states)

# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------
class MonsterA24Upgraded(nn.Module):
    def __init__(self, n_axes=8, dim=128):
        super().__init__()
        self.n_axes = n_axes
        self.dim = dim
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, n_axes * dim))
        self.residual = GradedResidual(dim)
        self.graph = GraphCoupling(n_axes)
        self.classifier = nn.Linear(n_axes * dim, 10)
        self.barrier_strength = nn.Parameter(torch.tensor(3.0))

    def encode_to_states(self, x):
        return self.encoder(x.view(x.size(0), -1)).view(x.size(0), self.n_axes, self.dim)

    def evolve_batch(self, states, class_ids):
        onehots = F.one_hot(class_ids, N_CLASSES).float()
        interaction = self.graph(states)
        strength = F.softplus(self.barrier_strength)
        new = []
        for a in range(self.n_axes):
            new.append(self.residual.step(states[:, a], onehots[:, a], class_ids[:, a],
                                          interaction[:, a], strength))
        return torch.stack(new, 1)

    def forward(self, x, class_ids):
        states = self.encode_to_states(x)
        states = self.evolve_batch(states, class_ids)
        return self.classifier(states.view(states.size(0), -1)), states

    def certificate_violation_rate(self, states, class_ids, tol=0.04):
        onehots = F.one_hot(class_ids, N_CLASSES).float()
        viol = total = 0
        B = states.size(0)
        for a in range(self.n_axes):
            pred = self.residual.predict_inner(states[:, a], onehots[:, a])
            targets = torch.tensor([NORTON[CLASSES[i]]["inner"] for i in class_ids[:, a]],
                                   device=states.device)
            viol += (pred - targets).abs().gt(tol).sum().item()
            total += B
        return viol / max(total, 1)

    def inner_mse(self, states, class_ids):
        onehots = F.one_hot(class_ids, N_CLASSES).float()
        mse = 0.0
        for a in range(self.n_axes):
            pred = self.residual.predict_inner(states[:, a], onehots[:, a])
            targets = torch.tensor([NORTON[CLASSES[i]]["inner"] for i in class_ids[:, a]],
                                   device=states.device)
            mse += F.mse_loss(pred, targets)
        return mse / self.n_axes

# ------------------------------------------------------------------
# Training + plot
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=256, shuffle=False
    )

    model = MonsterA24Upgraded(n_axes=8, dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss()

    def make_class_ids(B, A):
        return torch.randint(0, N_CLASSES, (B, A), device=device)

    epochs = 15
    curriculum = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 16.0, 16.0,
                  16.0, 16.0, 16.0, 16.0, 16.0]

    history = {"train_acc": [], "test_acc": [], "train_viol": [], "test_viol": []}

    print("=== Final Upgraded Monster A24 (15 epochs) ===")
    for epoch in range(epochs):
        reg = curriculum[epoch]
        print(f"Epoch {epoch} | inner-MSE coefficient: {reg}")
        model.train()
        total_acc = total_viol = n = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            class_ids = make_class_ids(data.size(0), model.n_axes)
            logits, states = model(data, class_ids)
            vrate = model.certificate_violation_rate(states, class_ids)
            mse = model.inner_mse(states, class_ids)
            loss = criterion(logits, target) + reg * mse + 2.0 * vrate
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pred = logits.argmax(1)
            total_acc += pred.eq(target).sum().item()
            total_viol += vrate * data.size(0)
            n += data.size(0)

            if batch_idx % 200 == 0:
                print(f"  [{batch_idx:4d}/{len(train_loader)}] "
                      f"Acc: {100.*pred.eq(target).sum().item()/data.size(0):.2f}% | Viol: {vrate:.3f}")

        model.eval()
        test_correct = test_viol = test_n = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                class_ids = make_class_ids(data.size(0), model.n_axes)
                logits, states = model(data, class_ids)
                test_correct += logits.argmax(1).eq(target).sum().item()
                test_viol += model.certificate_violation_rate(states, class_ids) * data.size(0)
                test_n += data.size(0)

        train_acc = 100. * total_acc / n
        train_viol = total_viol / n
        test_acc = 100. * test_correct / test_n
        test_viol = test_viol / test_n

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_viol"].append(train_viol)
        history["test_viol"].append(test_viol)

        print(f"=== Epoch {epoch} Summary ===")
        print(f"Train Acc: {train_acc:.2f}% | Avg Viol: {train_viol:.4f}")
        print(f"Test Acc: {test_acc:.2f}% | Test Viol: {test_viol:.4f}")

    torch.save(model.state_dict(), "monster_a24_final.pt")
    print("Saved monster_a24_final.pt")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    epochs_range = list(range(epochs))
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_acc = "#1f77b4"
    color_viol = "#d62728"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)", color=color_acc)
    ax1.plot(epochs_range, history["train_acc"], "o-", color=color_acc, label="Train Acc")
    ax1.plot(epochs_range, history["test_acc"], "s--", color=color_acc, alpha=0.7, label="Test Acc")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(90, 100)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Violation Rate", color=color_viol)
    ax2.plot(epochs_range, history["train_viol"], "o-", color=color_viol, label="Train Viol")
    ax2.plot(epochs_range, history["test_viol"], "s--", color=color_viol, alpha=0.7, label="Test Viol")
    ax2.tick_params(axis="y", labelcolor=color_viol)
    ax2.set_ylim(0, 0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Monster A24 Final — Accuracy vs Violation Rate")
    fig.tight_layout()
    plt.savefig("monster_a24_final_evolution.png", dpi=150)
    print("Plot saved to monster_a24_final_evolution.png")
    plt.show()

if __name__ == "__main__":
    main()
