import torch
from app.models.dqn_model import MultiStockDQN

def load_dqn_model(path: str) -> MultiStockDQN:
    model = MultiStockDQN()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except FileNotFoundError:
        print(f"[경고] '{path}' 파일이 없어 무작위 가중치 모델을 로딩합니다.")
    model.eval()
    return model
