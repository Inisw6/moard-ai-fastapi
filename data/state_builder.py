import torch
import torch.nn as nn

class GoalEmbedding(nn.Module):
    """
    테마 ID → goal embedding 벡터
    Meta-DQN의 conditioning input으로 사용됨
    """
    def __init__(self, num_themes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_themes, embedding_dim=embed_dim)

    def forward(self, theme_ids: torch.Tensor):
        """
        Parameters:
            theme_ids: [B] 정수 ID (LongTensor)
        Returns:
            goal_embeds: [B, embed_dim] 실수 벡터
        """
        return self.embedding(theme_ids)

class StateBuilder:
    """
    user_embed + goal_embed → Meta-DQN 입력 상태 벡터 생성기
    """
    def __init__(self, goal_embedding_layer: GoalEmbedding):
        self.goal_embedding_layer = goal_embedding_layer

    def build_state(self, user_embeds: torch.Tensor, theme_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            user_embeds: [B, user_embed_dim]
            theme_ids: [B] or [B, 1]
        Returns:
            state: [B, user_embed_dim + goal_embed_dim]
        """
        goal_embeds = self.goal_embedding_layer(theme_ids)     # [B, goal_embed_dim]
        state = torch.cat([user_embeds, goal_embeds], dim=1)   # [B, total_dim]
        return state
