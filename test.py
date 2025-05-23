import numpy as np
import json

# 후보 콘텐츠 벡터 생성기
def generate_candidates(n=5):
    return [
        {
            "content_embed": np.random.randn(256).round(4).tolist(),
            "sentiment_id": int(np.random.choice([0, 1, 2]))
        }
        for _ in range(n)
    ]

# 전체 요청 구조 생성
def generate_recommend_full_all_input():
    return {
        "user_embed": np.random.randn(256).round(4).tolist(),
        "theme_id": int(np.random.randint(0, 100)),
        "blog_candidates": generate_candidates(5),
        "youtube_candidates": generate_candidates(5),
        "news_candidates": generate_candidates(5)
    }
print(json.dumps(generate_recommend_full_all_input())
)