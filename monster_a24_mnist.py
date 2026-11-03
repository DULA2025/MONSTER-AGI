# monster_a24_mnist.py
"""
Monster / A24 Liquid Architecture — MNIST Training Test
Encodes images into axis states, evolves under certified dynamics, classifies.
Tracks both accuracy and algebraic certificate violation rate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

# ----------------------------------------------------------------------
# Certified Norton spectrum (unchanged)
# ----------------------------------------------------------------------
NORTON: Dict[str, Dict[str, float]] = {
    "1A": {"inner": 1.0,      "radius": 2.2},
    "2A": {"inner": 0.25,     "radius": 5.5},
    "2B": {"inner": 0.0,      "radius": 8.2},
    "3A": {"inner": 13/256,   "radius": 10.6},
    "3C": {"inner": 1/64,     "radius": 12.8},
    "4A": {"inner": 1/32,     "radius": 15.0},
    "4B": {"inner": 1/128,    "radius": 17.2},
    "5A": {"inner": 3/128,    "radius": 19.5},
    "6A": {"inner": 5/256,    "radius": 21.8},
}
CLASS_LIST = list(NORTON.keys())

EIGENSPACE_DIMS = [4, 16, 32, 32]
TOTAL_DIM = sum(EIGENSPACE_DIMS)
N_AXES = 8   # smaller for MNIST speed; increase if desired

@dataclass
class AxisState:
    class_id: str
    continuous: torch.Tensor
    axis_id: int = 0

class CertificateInterface:
    def __init__(self, tol: float = 0.04):
        self.tol = tol
        self.history: List[Tuple[str, float, float, bool]] = []

    def check(self, class_id: str, observed: float) -> bool:
        target = NORTON[class_id]["inner"]
        ok = abs(observed - target) < self.tol
        self.history.append((class_id, observed, target, ok))
        return ok

    def violation_rate(self, last_n: Optional[int] = None) -> float:
        records = self.history[-last_n:] if last_n else self.history
        if not records:
            return 0.0
        return sum(1 for *_, ok in records if not ok) / len(records)

    def reset(self):
        self.history.clear()

class PrimeInertiaControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_barrier = nn.Parameter(torch.tensor(0.5))
        self.raw_interact = nn.Parameter(torch.tensor(0.0))

    @property
    def barrier_scale(self) -> torch.Tensor:
        return 0.5 + F.softplus(self.raw_barrier)

    @property
    def interact_scale(self) -> torch.Tensor:
        return 0.03 + 0.2 * torch.sigmoid(self.raw_interact)

class GraphCoupling(nn.Module):
    def __init__(self, max_degree: int = 3):
        super().__init__()
        self.max_degree = max_degree
        self.edge_strength = nn.Parameter(torch.tensor(0.08))

    def build_edges(self, n: int) -> List[Tuple[int, int]]:
        edges = [(i, (i + 1) % n) for i in range(n)]
        if n > 4:
            edges += [(i, (i + n // 2) % n) for i in range(0, n, 2)]
        return edges[:n * self.max_degree // 2]

    def forces(self, states: List[AxisState], interact_scale: float) -> List[torch.Tensor]:
        n = len(states)
        forces = [torch.zeros(TOTAL_DIM, device=states[0].continuous.device) for _ in range(n)]
        edges = self.build_edges(n)
        strength = interact_scale * self.edge_strength.abs()
        for i, j in edges:
            si, sj = states[i], states[j]
            pair = 0.5 * (NORTON[si.class_id]["inner"] + NORTON[sj.class_id]["inner"])
            dir_ij = sj.continuous / (sj.continuous.norm() + 1e-8)
            dir_ji = si.continuous / (si.continuous.norm() + 1e-8)
            forces[i] = forces[i] + strength * pair * dir_ij
            forces[j] = forces[j] + strength * pair * dir_ji
        return forces

class LiquidResidual(nn.Module):
    def __init__(self, dim=TOTAL_DIM):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
            nn.Linear(96, dim)
        )
        self.tau_net = nn.Linear(dim + 9, 1)
        self.base_strength = nn.Parameter(torch.tensor(2.4))
        self.inner_readout = nn.Sequential(
            nn.Linear(dim + 9, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def barrier_force(self, x, class_id, strength):
        r_target = NORTON[class_id]["radius"]
        r = x.norm(dim=-1, keepdim=True)
        delta = r - r_target
        return strength * delta * (-x / (r + 1e-8))

    def step(self, x, class_onehot, class_id, interaction_force, strength, dt=0.08):
        combined = torch.cat([x, class_onehot], -1)
        tau = F.softplus(self.tau_net(combined)) + 0.1
        dx = -x / tau + self.f(x) + self.barrier_force(x, class_id, strength) + interaction_force
        return x + dt * dx

    def predict_inner(self, x, class_onehot):
        return self.inner_readout(torch.cat([x, class_onehot], -1)).squeeze(-1)

class MonsterA24MNIST(nn.Module):
    def __init__(self, n_axes: int = N_AXES, evolve_steps: int = 6):
        super().__init__()
        self.n_axes = n_axes
        self.evolve_steps = evolve_steps
        self.residual = LiquidResidual()
        self.class_embed = nn.Embedding(9, 9)
        self.control = PrimeInertiaControl()
        self.graph = GraphCoupling()
        self.cert = CertificateInterface(tol=0.04)

        # Image encoder → initial axis states
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, n_axes * TOTAL_DIM)
        )
        # Classifier from final axis states
        self.classifier = nn.Sequential(
            nn.Linear(n_axes * TOTAL_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def _onehot(self, class_id: str, device):
        idx = CLASS_LIST.index(class_id)
        return self.class_embed(torch.tensor(idx, device=device))

    def encode_to_states(self, x: torch.Tensor) -> List[List[AxisState]]:
        """Batch of images → list of lists of AxisState."""
        B = x.size(0)
        device = x.device
        encoded = self.encoder(x).view(B, self.n_axes, TOTAL_DIM)
        batch_states = []
        for b in range(B):
            states = []
            for i in range(self.n_axes):
                # Assign classes in round-robin for structure
                class_id = CLASS_LIST[i % 9]
                cont = encoded[b, i]
                # Softly normalize toward certified radius
                r_target = NORTON[class_id]["radius"]
                cont = cont / (cont.norm() + 1e-8) * r_target
                states.append(AxisState(class_id, cont, i))
            batch_states.append(states)
        return batch_states

    def evolve_batch(self, batch_states: List[List[AxisState]]) -> List[List[AxisState]]:
        device = batch_states[0][0].continuous.device
        for _ in range(self.evolve_steps):
            new_batch = []
            for states in batch_states:
                forces = self.graph.forces(states, self.control.interact_scale.item())
                strength = (self.control.barrier_scale * self.residual.base_strength.abs()).item()
                new_states = []
                for s, f in zip(states, forces):
                    onehot = self._onehot(s.class_id, device)
                    x_new = self.residual.step(s.continuous, onehot, s.class_id, f, strength)
                    new_states.append(AxisState(s.class_id, x_new, s.axis_id))
                new_batch.append(new_states)
            batch_states = new_batch
        return batch_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_states = self.encode_to_states(x)
        batch_states = self.evolve_batch(batch_states)
        # Flatten final states for classification
        flats = []
        for states in batch_states:
            flat = torch.cat([s.continuous for s in states])
            flats.append(flat)
        features = torch.stack(flats)
        return self.classifier(features)

    def certificate_violation_rate(self, batch_states: List[List[AxisState]]) -> float:
        self.cert.reset()
        for states in batch_states:
            for s in states:
                onehot = self._onehot(s.class_id, s.continuous.device)
                pred = self.residual.predict_inner(s.continuous, onehot).item()
                self.cert.check(s.class_id, pred)
        return self.cert.violation_rate()

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    model = MonsterA24MNIST(n_axes=N_AXES, evolve_steps=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)

    # monster_a24_mnist.py
"""
Monster / A24 Liquid Architecture — MNIST Training Test
Encodes images into axis states, evolves under certified dynamics, classifies.
Tracks both accuracy and algebraic certificate violation rate.
Stronger invariant regularizer + radius penalty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ----------------------------------------------------------------------
# Certified Norton spectrum (unchanged)
# ----------------------------------------------------------------------
NORTON: Dict[str, Dict[str, float]] = {
    "1A": {"inner": 1.0,      "radius": 2.2},
    "2A": {"inner": 0.25,     "radius": 5.5},
    "2B": {"inner": 0.0,      "radius": 8.2},
    "3A": {"inner": 13/256,   "radius": 10.6},
    "3C": {"inner": 1/64,     "radius": 12.8},
    "4A": {"inner": 1/32,     "radius": 15.0},
    "4B": {"inner": 1/128,    "radius": 17.2},
    "5A": {"inner": 3/128,    "radius": 19.5},
    "6A": {"inner": 5/256,    "radius": 21.8},
}
CLASS_LIST = list(NORTON.keys())

EIGENSPACE_DIMS = [4, 16, 32, 32]
TOTAL_DIM = sum(EIGENSPACE_DIMS)
N_AXES = 8

@dataclass
class AxisState:
    class_id: str
    continuous: torch.Tensor
    axis_id: int = 0

class CertificateInterface:
    def __init__(self, tol: float = 0.04):
        self.tol = tol
        self.history: List[Tuple[str, float, float, bool]] = []

    def check(self, class_id: str, observed: float) -> bool:
        target = NORTON[class_id]["inner"]
        ok = abs(observed - target) < self.tol
        self.history.append((class_id, observed, target, ok))
        return ok

    def violation_rate(self, last_n: Optional[int] = None) -> float:
        records = self.history[-last_n:] if last_n else self.history
        if not records:
            return 0.0
        return sum(1 for *_, ok in records if not ok) / len(records)

    def reset(self):
        self.history.clear()

class PrimeInertiaControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_barrier = nn.Parameter(torch.tensor(0.5))
        self.raw_interact = nn.Parameter(torch.tensor(0.0))

    @property
    def barrier_scale(self) -> torch.Tensor:
        return 0.5 + F.softplus(self.raw_barrier)

    @property
    def interact_scale(self) -> torch.Tensor:
        return 0.03 + 0.2 * torch.sigmoid(self.raw_interact)

class GraphCoupling(nn.Module):
    def __init__(self, max_degree: int = 3):
        super().__init__()
        self.max_degree = max_degree
        self.edge_strength = nn.Parameter(torch.tensor(0.08))

    def build_edges(self, n: int) -> List[Tuple[int, int]]:
        edges = [(i, (i + 1) % n) for i in range(n)]
        if n > 4:
            edges += [(i, (i + n // 2) % n) for i in range(0, n, 2)]
        return edges[:n * self.max_degree // 2]

    def forces(self, states: List[AxisState], interact_scale: float) -> List[torch.Tensor]:
        n = len(states)
        forces = [torch.zeros(TOTAL_DIM, device=states[0].continuous.device) for _ in range(n)]
        edges = self.build_edges(n)
        strength = interact_scale * self.edge_strength.abs()
        for i, j in edges:
            si, sj = states[i], states[j]
            pair = 0.5 * (NORTON[si.class_id]["inner"] + NORTON[sj.class_id]["inner"])
            dir_ij = sj.continuous / (sj.continuous.norm() + 1e-8)
            dir_ji = si.continuous / (si.continuous.norm() + 1e-8)
            forces[i] = forces[i] + strength * pair * dir_ij
            forces[j] = forces[j] + strength * pair * dir_ji
        return forces

class LiquidResidual(nn.Module):
    def __init__(self, dim=TOTAL_DIM):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
            nn.Linear(96, dim)
        )
        self.tau_net = nn.Linear(dim + 9, 1)
        self.base_strength = nn.Parameter(torch.tensor(2.4))
        self.inner_readout = nn.Sequential(
            nn.Linear(dim + 9, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def barrier_force(self, x, class_id, strength):
        r_target = NORTON[class_id]["radius"]
        r = x.norm(dim=-1, keepdim=True)
        delta = r - r_target
        return strength * delta * (-x / (r + 1e-8))

    def step(self, x, class_onehot, class_id, interaction_force, strength, dt=0.08):
        combined = torch.cat([x, class_onehot], -1)
        tau = F.softplus(self.tau_net(combined)) + 0.1
        dx = -x / tau + self.f(x) + self.barrier_force(x, class_id, strength) + interaction_force
        return x + dt * dx

    def predict_inner(self, x, class_onehot):
        return self.inner_readout(torch.cat([x, class_onehot], -1)).squeeze(-1)

class MonsterA24MNIST(nn.Module):
    def __init__(self, n_axes: int = N_AXES, evolve_steps: int = 6):
        super().__init__()
        self.n_axes = n_axes
        self.evolve_steps = evolve_steps
        self.residual = LiquidResidual()
        self.class_embed = nn.Embedding(9, 9)
        self.control = PrimeInertiaControl()
        self.graph = GraphCoupling()
        self.cert = CertificateInterface(tol=0.04)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, n_axes * TOTAL_DIM)
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_axes * TOTAL_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def _onehot(self, class_id: str, device):
        idx = CLASS_LIST.index(class_id)
        return self.class_embed(torch.tensor(idx, device=device))

    def encode_to_states(self, x: torch.Tensor) -> List[List[AxisState]]:
        B = x.size(0)
        device = x.device
        encoded = self.encoder(x).view(B, self.n_axes, TOTAL_DIM)
        batch_states = []
        for b in range(B):
            states = []
            for i in range(self.n_axes):
                class_id = CLASS_LIST[i % 9]
                cont = encoded[b, i]
                r_target = NORTON[class_id]["radius"]
                cont = cont / (cont.norm() + 1e-8) * r_target
                states.append(AxisState(class_id, cont, i))
            batch_states.append(states)
        return batch_states

    def evolve_batch(self, batch_states: List[List[AxisState]]) -> List[List[AxisState]]:
        device = batch_states[0][0].continuous.device
        for _ in range(self.evolve_steps):
            new_batch = []
            for states in batch_states:
                forces = self.graph.forces(states, self.control.interact_scale.item())
                strength = (self.control.barrier_scale * self.residual.base_strength.abs()).item()
                new_states = []
                for s, f in zip(states, forces):
                    onehot = self._onehot(s.class_id, device)
                    x_new = self.residual.step(s.continuous, onehot, s.class_id, f, strength)
                    new_states.append(AxisState(s.class_id, x_new, s.axis_id))
                new_batch.append(new_states)
            batch_states = new_batch
        return batch_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_states = self.encode_to_states(x)
        batch_states = self.evolve_batch(batch_states)
        flats = [torch.cat([s.continuous for s in states]) for states in batch_states]
        features = torch.stack(flats)
        return self.classifier(features)

    def certificate_violation_rate(self, batch_states: List[List[AxisState]]) -> float:
        self.cert.reset()
        for states in batch_states:
            for s in states:
                onehot = self._onehot(s.class_id, s.continuous.device)
                pred = self.residual.predict_inner(s.continuous, onehot).item()
                self.cert.check(s.class_id, pred)
        return self.cert.violation_rate()

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    model = MonsterA24MNIST(n_axes=N_AXES, evolve_steps=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)

    print("=== MNIST Training with Monster A24 Backbone ===")
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_viol = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            output = model(data)
            loss_cls = F.cross_entropy(output, target)

            # Certificate + radius regularizer
            with torch.no_grad():
                batch_states = model.encode_to_states(data)
                batch_states = model.evolve_batch(batch_states)
                vrate = model.certificate_violation_rate(batch_states)

                # Explicit radius penalty
                radius_penalty = 0.0
                n_states = 0
                for states in batch_states:
                    for s in states:
                        target_r = NORTON[s.class_id]["radius"]
                        radius_penalty += (s.continuous.norm() - target_r).abs().item()
                        n_states += 1
                radius_penalty = radius_penalty / max(n_states, 1)

            loss = loss_cls + 8.0 * vrate + 3.0 * radius_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            epoch_viol.append(vrate)

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | "
                      f"Viol: {vrate:.3f} | Rad: {radius_penalty:.3f}")

        scheduler.step()
        train_acc = 100. * correct / total
        avg_viol = sum(epoch_viol) / len(epoch_viol)
        print(f"=== Epoch {epoch} Summary ===")
        print(f"Train Acc: {train_acc:.2f}% | Avg Viol: {avg_viol:.3f} | "
              f"Avg Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        viol_rates = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
                states = model.encode_to_states(data)
                states = model.evolve_batch(states)
                viol_rates.append(model.certificate_violation_rate(states))

        test_acc = 100. * test_correct / test_total
        avg_test_viol = sum(viol_rates) / len(viol_rates)
        print(f"Test Acc: {test_acc:.2f}% | Test Viol: {avg_test_viol:.4f}\n")

if __name__ == "__main__":
    main()
