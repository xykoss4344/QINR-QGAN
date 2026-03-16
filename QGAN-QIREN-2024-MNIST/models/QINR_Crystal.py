
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

        torch_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3), "weights2": (2, in_features, 3)}

        self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")

        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
            
        # Prevent CUDA context clashes by invoking the quantum simulator explicitly on CPU
        curr_device = x.device
        x_cpu = x.to('cpu')
        out_cpu = self.qnn(x_cpu)
        out = out_cpu.to(curr_device)
        
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
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True):
            super().__init__()

            self.net = []
            self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1))

            for i in range(hidden_layers):
                self.net.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))

            if outermost_linear:
                # We want final output to be out_features (90)
                # But here we have an intermediate linear layer
                final_linear = nn.Linear(hidden_features, 128)
            else:
                final_linear = HybridLayer(hidden_features, out_features, spectrum_layer, use_noise)

            final_linear_1 = nn.Linear(128, 512)
            final_linear_2 = nn.Linear(512, 256)
            final_linear_3 = nn.Linear(256, out_features) # Direct output size
            self.net.append(final_linear)
            self.net.append(final_linear_1)
            self.net.append(final_linear_2)
            self.net.append(final_linear_3)
            self.net = nn.Sequential(*self.net)

        def forward(self, coords):
            coords = coords.clone().detach().requires_grad_(True)
            output = self.net(coords)
            # output shape should be (batch, out_features)
            return output
