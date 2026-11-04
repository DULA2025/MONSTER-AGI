from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# Paste the same model classes (GradedResidual, GraphCoupling, MonsterA24Upgraded)
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

# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MonsterA24Upgraded().to(device)
model.load_state_dict(torch.load("monster_a24_final.pt", map_location=device, weights_only=True))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    pixels = np.array(data["pixels"], dtype=np.float32).reshape(1, 1, 28, 28)
    # MNIST normalization
    pixels = (pixels - 0.1307) / 0.3081
    x = torch.from_numpy(pixels).to(device)
    class_ids = torch.randint(0, N_CLASSES, (1, model.n_axes), device=device)
    with torch.no_grad():
        logits, _ = model(x, class_ids)
        probs = torch.softmax(logits, dim=1)
        digit = int(probs.argmax(1).item())
        confidence = float(probs[0, digit].item())
    return jsonify({"digit": digit, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)