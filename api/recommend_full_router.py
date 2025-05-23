from fastapi import APIRouter
from pydantic import BaseModel
import torch
from model.meta_dqn import MetaDQN
from model.content_dqn import ContentDQN
from model.candidate_encoder import CandidateEncoder
from data.state_builder import GoalEmbedding, StateBuilder

router = APIRouter()

# 설정
STATE_DIM = 320
USER_DIM = 256
GOAL_DIM = 64
ACTION_DIM = 27
NUM_THEMES = 100
K = 3  # Top-K

# 상위 MetaDQN 로딩
goal_embed_layer = GoalEmbedding(num_themes=NUM_THEMES, embed_dim=GOAL_DIM)
goal_embed_layer.eval()
state_builder = StateBuilder(goal_embed_layer)
meta_model = MetaDQN(state_dim=STATE_DIM, action_dim=ACTION_DIM)
meta_model.load_state_dict(torch.load("models/meta_dqn.pth", map_location="cpu"))
meta_model.eval()

# 하위 ContentDQNs 로딩
candidate_encoder = CandidateEncoder()
STATE_DIM_FULL = USER_DIM + GOAL_DIM + candidate_encoder.output_dim
content_models = {t: ContentDQN(STATE_DIM_FULL) for t in ["YouTube","Blog","News"]}
for t, m in content_models.items():
    m.load_state_dict(torch.load(f"models/{t.lower()}_dqn.pth", map_location="cpu"))
    m.eval()

# 액션 매핑 테이블
content_types = ["YouTube","Blog","News"]
scopes = ["단일종목","테마","시황"]
formats = ["요약형","보고서형","속보형"]
ACTION_ID_TO_NAME = {i: f"{ct}_{sc}_{fm}" for i,(ct,sc,fm) in 
    enumerate([(ct,sc,fm) for ct in content_types for sc in scopes for fm in formats])}

# 요청/응답
class CandidateItem(BaseModel):
    content_embed: list[float]
    sentiment_id: int

class RecommendPerType(BaseModel):
    type: str
    action_name: str
    top_k_indices: list[int]
    q_values: list[float]

class RecommendFullAllRequest(BaseModel):
    user_embed: list[float]
    theme_id: int
    blog_candidates: list[CandidateItem]
    youtube_candidates: list[CandidateItem]
    news_candidates: list[CandidateItem]

class RecommendFullAllResponse(BaseModel):
    recommendations: list[RecommendPerType]

@router.post("/recommend_full_all", response_model=RecommendFullAllResponse)
def recommend_full_all(req: RecommendFullAllRequest):
    # MetaDQN 실행
    user_t = torch.tensor([req.user_embed], dtype=torch.float32)
    theme_t = torch.tensor([req.theme_id], dtype=torch.long)
    with torch.no_grad():
        meta_state = state_builder.build_state(user_t, theme_t)
        meta_q = meta_model(meta_state).squeeze(0)

    results = []
    # 3개 유형별로 Top-K 및 메타 액션 이름 계산
    for i, (t, cand_list) in enumerate(zip(content_types, [req.youtube_candidates, req.blog_candidates, req.news_candidates])):
        # Meta action local to type\        
        start = i * (ACTION_DIM // len(content_types))
        end = start + (ACTION_DIM // len(content_types))
        segment = meta_q[start:end]
        rel_idx = torch.argmax(segment).item()
        abs_idx = start + rel_idx
        action_name = ACTION_ID_TO_NAME[abs_idx]

        # 하위 후보 Q값
        state_list = []
        for c in cand_list:
            ce = torch.tensor(c.content_embed, dtype=torch.float32)
            si = torch.tensor(c.sentiment_id, dtype=torch.long)
            cv = candidate_encoder(ce.unsqueeze(0), si.unsqueeze(0)).squeeze(0)
            state_list.append(torch.cat([user_t.squeeze(0), goal_embed_layer(theme_t).squeeze(0), cv], dim=-1))
        sb = torch.stack(state_list)
        with torch.no_grad():
            qv = content_models[t](sb).squeeze(1)
            topk = torch.topk(qv, k=min(K, qv.size(0)))

        results.append(RecommendPerType(
            type=t,
            action_name=action_name,
            top_k_indices=topk.indices.tolist(),
            q_values=qv.tolist()
        ))

    return RecommendFullAllResponse(recommendations=results)
