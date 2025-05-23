import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from model.content_dqn import ContentDQN
from model.candidate_encoder import CandidateEncoder

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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

# Dummy Dataset 생성기
def generate_dummy_blog_dataset(num_samples=500, user_dim=256, goal_dim=64, content_dim=256):
    dataset = []
    for _ in range(num_samples):
        dataset.append({
            "user_embed": np.random.randn(user_dim).astype(np.float32),
            "goal_embed": np.random.randn(goal_dim).astype(np.float32),
            "content_embed": np.random.randn(content_dim).astype(np.float32),
            "sentiment_id": np.random.randint(0, 3),  # 0=부정, 1=중립, 2=긍정
            "reward": float(np.round(np.random.rand(), 2)),
            "done": bool(np.random.rand() < 0.1)
        })
    return dataset

# 학습 함수
def train_blog_dqn(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_encoder = CandidateEncoder().to(device)
    state_dim = config["user_dim"] + config["goal_dim"] + candidate_encoder.output_dim
    q_net = ContentDQN(state_dim).to(device)
    target_net = ContentDQN(state_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])

    for row in dataset:
        user_embed = torch.tensor(row["user_embed"], dtype=torch.float32)
        goal_embed = torch.tensor(row["goal_embed"], dtype=torch.float32)
        content_embed = torch.tensor(row["content_embed"], dtype=torch.float32)
        sentiment_id = torch.tensor(row["sentiment_id"], dtype=torch.long)
        reward = row["reward"]
        done = row["done"]

        with torch.no_grad():
            content_vector = candidate_encoder(content_embed.unsqueeze(0), sentiment_id.unsqueeze(0)).squeeze(0)
            state = torch.cat([user_embed, goal_embed, content_vector], dim=-1)
            next_state = state.clone()

        buffer.add(state, 0, reward, next_state, done)

    for epoch in range(config["epochs"]):
        if len(buffer.buffer) < config["batch_size"]:
            continue

        states, actions, rewards, next_states, dones = buffer.sample(config["batch_size"])
        states = states.to(device)
        next_states = next_states.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        q_values = q_net(states).squeeze(1)
        with torch.no_grad():
            max_q_next = target_net(next_states).squeeze(1)
            targets = rewards + config["gamma"] * max_q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % config["target_update"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        if epoch % config["log_interval"] == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    torch.save(q_net.state_dict(), "models/blog_dqn.pth")
    print("✅ Blog DQN 모델 저장 완료: models/blog_dqn.pth")

# 실행 엔트리포인트
def main():
    config = {
        "user_dim": 256,
        "goal_dim": 64,
        "lr": 0.001,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 32,
        "epochs": 200,
        "target_update": 50,
        "log_interval": 20
    }
    dataset = generate_dummy_blog_dataset(num_samples=500)
    train_blog_dqn(config, dataset)

if __name__ == "__main__":
    main()
