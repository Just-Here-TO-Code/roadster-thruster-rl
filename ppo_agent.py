import json
import math
import numpy as np

# Normalization scaling vector for the 15-dimensional observation space.
# It brings positions, velocities, and angular rates into [-1, 1] so neural network
# gradients stay balanced and training converges smoothly.
OBS_SCALE = np.array([
    1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0,    # x, y, z positions
    1.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0,    # vx, vy, vz linear velocities
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,          # sin and cos of roll, pitch, yaw
    1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0        # wx, wy, wz angular rates
], dtype=np.float32)


class DenseLayer:
    """
    A single dense layer in the neural network.
    Performs forward pass z = x @ W + b and backpropagates gradients.
    """

    def __init__(self, in_dim, out_dim, activation="tanh"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        # Xavier initialization
        limit = math.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * limit
        self.b = np.zeros(out_dim, dtype=np.float32)

        # Allocate gradient buffers
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.last_input = None
        self.last_out = None

    def forward(self, x):
        """
        Forward computation with either tanh, sigmoid, or linear activation.
        """
        self.last_input = np.atleast_2d(x).astype(np.float32)
        z = self.last_input @ self.W + self.b

        if self.activation == "tanh":
            self.last_out = np.tanh(z)
        elif self.activation == "sigmoid":
            # Stable sigmoid implementation
            self.last_out = np.where(
                z >= 0,
                1.0 / (1.0 + np.exp(-z)),
                np.exp(z) / (1.0 + np.exp(z))
            )
        else:
            self.last_out = z

        return self.last_out

    def backward(self, d_out):
        """
        Backpropagation through activation function and matrix multiplication.
        Uses in-place slice assignment [:] so optimizer pointers stay linked.
        """
        if self.activation == "tanh":
            d_z = d_out * (1.0 - self.last_out**2)
        elif self.activation == "sigmoid":
            d_z = d_out * self.last_out * (1.0 - self.last_out)
        else:
            d_z = d_out

        # In-place gradient updates
        self.dW[:] = self.last_input.T @ d_z
        self.db[:] = np.sum(d_z, axis=0)

        # Propagate error backward to the previous layer
        d_in = d_z @ self.W.T
        return d_in


class MLP:
    """
    Multi-Layer Perceptron connecting layers in sequence.
    """

    def __init__(self, dims, activations):
        self.layers = []
        for i in range(len(dims) - 1):
            layer = DenseLayer(dims[i], dims[i + 1], activations[i])
            self.layers.append(layer)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_out):
        grad = d_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class PPOAgent:
    """
    Continuous control Actor-Critic agent using Proximal Policy Optimization (PPO).
    
    Actor network: 15 -> 64 -> 64 -> 4 (Sigmoid output for 4 thruster throttles)
    Critic network: 15 -> 64 -> 64 -> 1 (Linear output for state value estimate)
    """

    def __init__(self, state_dim=15, action_dim=4, lr=0.003):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        # Actor maps normalized 15D state to 4 continuous throttles in [0, 1]
        self.actor = MLP([state_dim, 64, 64, action_dim], ["tanh", "tanh", "sigmoid"])

        # Critic maps normalized 15D state to expected return V(s)
        self.critic = MLP([state_dim, 64, 64, 1], ["tanh", "tanh", "linear"])

        # Exploration noise standard deviation
        self.log_std = np.full(action_dim, -1.2, dtype=np.float32)

        # Adam momentum buffers for Actor
        self.mW = [np.zeros_like(l.W) for l in self.actor.layers]
        self.vW = [np.zeros_like(l.W) for l in self.actor.layers]
        self.mb = [np.zeros_like(l.b) for l in self.actor.layers]
        self.vb = [np.zeros_like(l.b) for l in self.actor.layers]

        # Adam momentum buffers for Critic
        self.c_mW = [np.zeros_like(l.W) for l in self.critic.layers]
        self.c_vW = [np.zeros_like(l.W) for l in self.critic.layers]
        self.c_mb = [np.zeros_like(l.b) for l in self.critic.layers]
        self.c_vb = [np.zeros_like(l.b) for l in self.critic.layers]

        self.step_count = 0

    def normalize_obs(self, obs):
        """
        Scales the observation vector into [-1, 1] range.
        """
        return obs * OBS_SCALE

    def get_action(self, obs, deterministic=False):
        """
        Computes action from the policy network.
        """
        norm_obs = self.normalize_obs(obs)
        mu = self.actor.forward(norm_obs)[0]
        val = self.critic.forward(norm_obs)[0, 0]

        if deterministic:
            action = np.clip(mu, 0.0, 1.0)
            log_prob = 0.0
        else:
            std = np.exp(self.log_std)
            noise = np.random.randn(self.action_dim).astype(np.float32)
            raw_action = mu + std * noise
            action = np.clip(raw_action, 0.0, 1.0)

            # Gaussian log-likelihood
            var = std**2
            log_prob = -0.5 * np.sum(
                ((action - mu)**2 / (var + 1e-8)) + 2.0 * self.log_std + math.log(2.0 * math.pi)
            )

        return action, float(log_prob), float(val)

    def train_imitation_step(self, states, target_actions):
        """
        Supervised imitation learning step to prime policy weights.
        """
        self.step_count += 1
        n = len(states)
        norm_states = states * OBS_SCALE

        preds = self.actor.forward(norm_states)
        loss = 0.5 * np.mean((preds - target_actions)**2)

        # Gradient of MSE with respect to predictions
        d_preds = (preds - target_actions) / n
        self.actor.backward(d_preds)

        # Adam parameter update
        for i, l in enumerate(self.actor.layers):
            self.mW[i] = 0.9 * self.mW[i] + 0.1 * l.dW
            self.vW[i] = 0.999 * self.vW[i] + 0.001 * (l.dW**2)
            self.mb[i] = 0.9 * self.mb[i] + 0.1 * l.db
            self.vb[i] = 0.999 * self.vb[i] + 0.001 * (l.db**2)

            l.W -= self.lr * self.mW[i] / (np.sqrt(self.vW[i]) + 1e-8)
            l.b -= self.lr * self.mb[i] / (np.sqrt(self.vb[i]) + 1e-8)

        return float(loss)

    def save(self, filepath):
        """
        Serializes trained weights and biases to a clean JSON file.
        """
        data = {
            "actor_weights": [l.W.tolist() for l in self.actor.layers],
            "actor_biases": [l.b.tolist() for l in self.actor.layers],
            "critic_weights": [l.W.tolist() for l in self.critic.layers],
            "critic_biases": [l.b.tolist() for l in self.critic.layers],
            "log_std": self.log_std.tolist()
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"Policy weights successfully saved to: {filepath}")

    def load(self, filepath):
        """
        Loads weights and biases from an existing JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        for i, l in enumerate(self.actor.layers):
            l.W = np.array(data["actor_weights"][i], dtype=np.float32)
            l.b = np.array(data["actor_biases"][i], dtype=np.float32)

        for i, l in enumerate(self.critic.layers):
            l.W = np.array(data["critic_weights"][i], dtype=np.float32)
            l.b = np.array(data["critic_biases"][i], dtype=np.float32)

        self.log_std = np.array(data["log_std"], dtype=np.float32)
        print(f"Policy weights successfully loaded from: {filepath}")
