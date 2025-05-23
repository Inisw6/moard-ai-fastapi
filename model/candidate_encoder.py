import torch
import torch.nn as nn
import torch.nn.functional as F

class CandidateEncoder(nn.Module):
    """
    콘텐츠 텍스트 + 감정 정보 → 콘텐츠 벡터 생성
    """
    def __init__(self, text_embed_dim=256, sentiment_embed_dim=16, use_dense_sentiment=True):
        super().__init__()
        self.use_dense_sentiment = use_dense_sentiment

        # 감정 임베딩: 0 = 부정, 1 = 중립, 2 = 긍정
        if self.use_dense_sentiment:
            self.sentiment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=sentiment_embed_dim)
            self.sentiment_dim = sentiment_embed_dim
        else:
            self.sentiment_dim = 3  # one-hot 방식

        self.output_dim = text_embed_dim + self.sentiment_dim

    def forward(self, text_embed: torch.Tensor, sentiment_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            text_embed: [B, text_embed_dim]
            sentiment_ids: [B]  (0~2: 부정/중립/긍정)
        Returns:
            content_vector: [B, output_dim] = concat(text_embed, sentiment_embed)
        """
        if self.use_dense_sentiment:
            sentiment_embed = self.sentiment_embedding(sentiment_ids)  # [B, 16]
        else:
            sentiment_embed = F.one_hot(sentiment_ids, num_classes=3).float()  # [B, 3]

        return torch.cat([text_embed, sentiment_embed], dim=1)
