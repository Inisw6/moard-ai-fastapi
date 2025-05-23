from fastapi import APIRouter
from pydantic import BaseModel
import torch
from model.meta_dqn import MetaDQN
from data.state_builder import GoalEmbedding, StateBuilder

router = APIRouter()

STATE_DIM = 320
USER_EMBED_DIM = 256
GOAL_EMBED_DIM = 64
ACTION_DIM = 27
NUM_THEMES = 100

# 액션 매핑 테이블 생성
content_types = ["YouTube", "Blog", "News"]
scopes = ["단일종목", "테마", "시황"]
formats = ["요약형", "보고서형", "속보형"]

ACTION_ID_TO_NAME = {
    i: f"{ct}_{sc}_{fm}"
    for i, (ct, sc, fm) in enumerate(
        [(ct, sc, fm) for ct in content_types for sc in scopes for fm in formats]
    )
}

# 모델 로딩
goal_embed_layer = GoalEmbedding(num_themes=NUM_THEMES, embed_dim=GOAL_EMBED_DIM)
goal_embed_layer.eval()
state_builder = StateBuilder(goal_embed_layer)

meta_dqn = MetaDQN(state_dim=STATE_DIM, action_dim=ACTION_DIM)
meta_dqn.load_state_dict(torch.load("models/meta_dqn.pth", map_location="cpu"))
meta_dqn.eval()

# 요청/응답 모델 정의
class RecommendRequest(BaseModel):
    user_embed: list[float]
    theme_id: int

class RecommendationItem(BaseModel):
    type: str
    action: int
    action_name: str
    q_value: float

class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]

@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    user_embed_tensor = torch.tensor([req.user_embed], dtype=torch.float32)
    theme_id_tensor = torch.tensor([req.theme_id], dtype=torch.long)

    with torch.no_grad():
        state = state_builder.build_state(user_embed_tensor, theme_id_tensor)
        q_values = meta_dqn(state).squeeze(0)  # [27]

        recommendations = []
        for i, content_type in enumerate(["YouTube", "Blog", "News"]):
            start = i * 9
            end = start + 9
            segment = q_values[start:end]
            rel_idx = torch.argmax(segment).item()
            abs_idx = start + rel_idx
            recommendations.append(RecommendationItem(
                type=content_type,
                action=abs_idx,
                action_name=ACTION_ID_TO_NAME[abs_idx],
                q_value=round(q_values[abs_idx].item(), 4)
            ))

    return RecommendResponse(recommendations=recommendations)