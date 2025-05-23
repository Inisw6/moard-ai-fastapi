import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from model.meta_dqn import MetaDQN
from data.state_builder import GoalEmbedding, StateBuilder


# 🔹 학습용 더미 데이터셋 생성 함수
def generate_dummy_training_dataset(num_samples=500, user_embed_dim=256, goal_embed_dim=64, action_dim=9):
    dataset = []
    for _ in range(num_samples):
        user_embed = np.random.randn(user_embed_dim).astype(np.float32)
        next_user_embed = np.random.randn(user_embed_dim).astype(np.float32)
        theme_id = int(np.random.randint(0, 100))
        next_theme_id = theme_id
        action = int(np.random.randint(0, action_dim))
        reward = float(np.round(np.random.rand(), 2))
        done = bool(np.random.rand() < 0.1)
        dataset.append({
            "user_embed": user_embed,
            "next_user_embed": next_user_embed,
            "theme_id": theme_id,
            "next_theme_id": next_theme_id,
            "action": action,
            "reward": reward,
            "done": done
        })
    return dataset


# 🔹 리플레이 버퍼
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*samples)
        return (
            torch.stack(s),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(s_),
            torch.tensor(d, dtype=torch.float32)
        )


# 🔹 Meta DQN 학습 함수
def train_meta_dqn(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = MetaDQN(config["state_dim"], config["action_dim"]).to(device)
    target_net = MetaDQN(config["state_dim"], config["action_dim"]).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    goal_embed_layer = GoalEmbedding(config["num_themes"], config["goal_embed_dim"]).to(device)
    goal_embed_layer.eval()  # 🔐 추론 전용으로 명시

    state_builder = StateBuilder(goal_embed_layer)

    optimizer = optim.Adam(q_net.parameters(), lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])

    for row in dataset:
        user_embed = torch.tensor(row["user_embed"], dtype=torch.float32)
        next_user_embed = torch.tensor(row["next_user_embed"], dtype=torch.float32)
        theme_id = torch.tensor([row["theme_id"]], dtype=torch.long)
        next_theme_id = torch.tensor([row["next_theme_id"]], dtype=torch.long)

        # 🔐 그래프 차단
        with torch.no_grad():
            s = state_builder.build_state(user_embed.unsqueeze(0), theme_id).squeeze(0).detach()
            s_ = state_builder.build_state(next_user_embed.unsqueeze(0), next_theme_id).squeeze(0).detach()

        buffer.add(s, row["action"], row["reward"], s_, row["done"])

    for epoch in range(config["epochs"]):
        if len(buffer.buffer) < config["batch_size"]:
            continue

        states, actions, rewards, next_states, dones = buffer.sample(config["batch_size"])

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        q_values = q_net(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_q_next = target_net(next_states).max(1)[0]
            target = rewards + config["gamma"] * max_q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_a, target.detach())  # 🔐 안전하게 학습

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % config["target_update"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        if epoch % config["log_interval"] == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    # 모델 저장
    torch.save(q_net.state_dict(), "models/meta_dqn.pth")
    print("✅ Meta DQN 모델 저장 완료: models/meta_dqn.pth")

    return q_net


# 🔹 실행 진입점
if __name__ == "__main__":
    config = {
        "state_dim": 320,
        "action_dim": 27,  # ← 변경됨
        "goal_embed_dim": 64,
        "num_themes": 100,
        "lr": 0.001,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 32,
        "epochs": 200,
        "target_update": 50,
        "log_interval": 20
    }

    dataset = generate_dummy_training_dataset(num_samples=500)
    train_meta_dqn(config, dataset)
