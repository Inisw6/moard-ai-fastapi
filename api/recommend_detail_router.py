from fastapi import APIRouter
from pydantic import BaseModel
import torch
from model.content_dqn import ContentDQN
from model.candidate_encoder import CandidateEncoder

router = APIRouter()

# 설정
USER_DIM = 256
GOAL_DIM = 64

# Candidate Encoder & Content DQNs 로딩
candidate_encoder = CandidateEncoder()
STATE_DIM = USER_DIM + GOAL_DIM + candidate_encoder.output_dim
models = {
    "Blog": ContentDQN(STATE_DIM),
    "YouTube": ContentDQN(STATE_DIM),
    "News": ContentDQN(STATE_DIM)
}
for key, model in models.items():
    model.load_state_dict(torch.load(f"models/{key.lower()}_dqn.pth", map_location="cpu"))
    model.eval()

# 요청/응답 스키마
class CandidateItem(BaseModel):
    content_embed: list[float]
    sentiment_id: int

class RecommendDetailRequest(BaseModel):
    user_embed: list[float]
    goal_embed: list[float]
    goal_type: str            # "Blog", "YouTube", "News"
    candidates: list[CandidateItem]

class RecommendDetailResponse(BaseModel):
    type: str                 # 콘텐츠 유형
    action: int               # Q 최대값 인덱스 (0~N-1)
    action_name: str          # 해당 유형의 Meta-DQN 액션 이름
    q_values: list[float]     # 모든 후보에 대한 Q값 리스트

@router.post("/recommend_detail", response_model=RecommendDetailResponse)
def recommend_detail(req: RecommendDetailRequest):
    # 하위 후보 Q값 계산
    user_tensor = torch.tensor(req.user_embed, dtype=torch.float32)
    goal_tensor = torch.tensor(req.goal_embed, dtype=torch.float32)
    state_list = []
    for c in req.candidates:
        ce = torch.tensor(c.content_embed, dtype=torch.float32)
        si = torch.tensor(c.sentiment_id, dtype=torch.long)
        cv = candidate_encoder(ce.unsqueeze(0), si.unsqueeze(0)).squeeze(0)
        state_list.append(torch.cat([user_tensor, goal_tensor, cv], dim=-1))
    states = torch.stack(state_list)

    model = models.get(req.goal_type)
    if not model:
        raise ValueError(f"Unknown goal_type: {req.goal_type}")

    with torch.no_grad():
        qv = model(states).squeeze(1)
        idx = torch.argmax(qv).item()

    # action_name은 해당 유형에서의 Meta-DQN 액션을 별도로 계산할 수 없음
    # placeholder로 유형_추천 사용하거나 삭제 가능
    action_name = f"{req.goal_type}_추천"

    return RecommendDetailResponse(
        type=req.goal_type,
        action=idx,
        action_name=action_name,
        q_values=qv.tolist()
    )