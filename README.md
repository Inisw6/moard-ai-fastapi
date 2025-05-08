# moard-api

## Supabase SQL
```
-- 1. 공통 콘텐츠 메타 테이블
CREATE TABLE content (
  id UUID PRIMARY KEY,
  type TEXT NOT NULL CHECK (type IN ('youtube', 'news', 'blog')),
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  keyword TEXT NOT NULL,
  summary TEXT,
  image_url TEXT,
  published_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT now()
);

-- 2. 유튜브 전용 테이블
CREATE TABLE youtube_content (
  content_id UUID PRIMARY KEY REFERENCES content(id),
  channel_title TEXT,
  view_count INTEGER,
  like_count INTEGER,
  comment_count INTEGER,
  duration_sec INTEGER
);

-- 3. 뉴스 전용 테이블
CREATE TABLE news_content (
  content_id UUID PRIMARY KEY REFERENCES content(id),
  press_name TEXT,
  is_opinion BOOLEAN DEFAULT false
);

-- 4. 블로그 전용 테이블
CREATE TABLE blog_content (
  content_id UUID PRIMARY KEY REFERENCES content(id),
  author TEXT,
  sentiment_score FLOAT,
  word_count INTEGER
);

-- 5. 사용자 로그 테이블
CREATE TABLE logs (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  client_id TEXT NOT NULL,
  content_id UUID REFERENCES content(id),
  event_type TEXT NOT NULL CHECK (event_type IN ('click', 'view', 'impression')),
  timestamp TIMESTAMP DEFAULT now(),
  duration_sec INTEGER,
  position INTEGER
);

-- 6. 추천 결과 기록 테이블
CREATE TABLE recommend_history (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  client_id TEXT NOT NULL,
  content_id UUID REFERENCES content(id),
  algorithm TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT now()
);

-- 7. 사용자 프로필 테이블 (pgvector 확장 필요)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE user_profile (
  client_id TEXT PRIMARY KEY,
  last_clicked_ids UUID[],
  preferred_type TEXT,
  keyword_history TEXT[],
  embedding vector(768)
);

-- 8. 키워드 태그 테이블
CREATE TABLE keyword_tags (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  content_id UUID REFERENCES content(id),
  tag TEXT
);

-- 9. 인덱스 추가
CREATE INDEX idx_content_keyword ON content(keyword);
CREATE INDEX idx_logs_client_id ON logs(client_id);
CREATE INDEX idx_logs_content_id ON logs(content_id);
CREATE INDEX idx_recommend_history_client_id ON recommend_history(client_id);

```