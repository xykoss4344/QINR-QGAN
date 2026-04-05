
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True, idx=0):
        super().__init__()
        self.idx = idx
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)

    def forward(self, x):
        x1 = self.clayer(x)
        out = self.qlayer(x1)
        return out


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layer, use_noise):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise

        def _circuit(inputs, weights1, weights2):
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)
                for j in range(self.in_features):
                    qml.RZ(inputs[..., j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        # PL 0.38+ unified default.qubit auto-detects PyTorch/CUDA interface.
        # backprop vectorises over the full batch — faster than adjoint for small qubit counts.
        ql_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3), "weights2": (2, in_features, 3)}

        self.qnode = qml.QNode(_circuit, ql_device, diff_method="backprop", interface="torch")

        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        out = self.qnn(x)
        return out.reshape(orgin_shape)

class PQWGAN_CC_Crystal():
    def __init__(self, input_dim_g, output_dim, input_dim_d, hidden_features, hidden_layers, spectrum_layer, use_noise, outermost_linear=True):
        self.output_dim = output_dim
        self.critic = self.ClassicalCritic(input_dim_d)
        self.generator = self.Hybridren(input_dim_g, hidden_features, hidden_layers, output_dim, spectrum_layer, use_noise, outermost_linear=True)

    class ClassicalCritic(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)

    class Hybridren(nn.Module):
        """
        Split-head generator:
          Shared quantum trunk  → 256-dim representation
          Cell head  (6 values) → 3 lengths (Sigmoid) + 3 angles (Sigmoid)
          Atom head (84 values) → 28 × 3 fractional positions (Sigmoid)
          Output: cat([cell, atom]) → 90-dim, matching the (30×3) crystal format.

        Keeping the heads separate ensures WGAN gradients can independently
        steer cell geometry vs atom positions, preventing cell-param collapse.
        """
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True):
            super().__init__()

            # ── Shared quantum trunk ──────────────────────────────────────────
            trunk = [HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1)]
            for i in range(hidden_layers):
                trunk.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))
            # Project quantum output to shared representation
            trunk += [
                nn.Linear(hidden_features, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
            ]
            self.trunk = nn.Sequential(*trunk)

            # ── Cell parameter head (6 outputs: 3 lengths + 3 angles) ────────
            # Both normalised to [0,1]: lengths/30, angles/180
            self.cell_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 6),
                nn.Sigmoid(),
            )

            # ── Atom position head (84 outputs: 28 atoms × 3 coords) ─────────
            # Fractional coordinates in [0,1]
            self.atom_head = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 84),
                nn.Sigmoid(),
            )

        def forward(self, coords):
            coords = coords.clone().detach().requires_grad_(True)
            shared = self.trunk(coords)          # (batch, 256)
            cell   = self.cell_head(shared)      # (batch, 6)
            atoms  = self.atom_head(shared)      # (batch, 84)
            return torch.cat([cell, atoms], dim=1)   # (batch, 90)
