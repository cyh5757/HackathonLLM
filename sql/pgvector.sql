CREATE EXTENSION IF NOT EXISTS vector;

-- PGVector 자동 생성 스키마 --
CREATE TABLE IF NOT EXISTS langchain_pg_collection
(
    uuid      uuid    NOT NULL PRIMARY KEY,
    name      varchar NOT NULL UNIQUE,
    cmetadata json
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding
(
    id            varchar NOT NULL PRIMARY KEY,
    collection_id uuid REFERENCES langchain_pg_collection ON DELETE CASCADE,
    embedding     vector,
    document      varchar,
    cmetadata     jsonb
);


CREATE UNIQUE INDEX IF NOT EXISTS ix_langchain_pg_embedding_id ON langchain_pg_embedding (id);
CREATE INDEX IF NOT EXISTS ix_cmetadata_gin ON langchain_pg_embedding USING gin (cmetadata jsonb_path_ops);

-- 수동으로 추가한 스키마 --
ALTER TABLE langchain_pg_embedding
    ALTER COLUMN embedding TYPE vector(1536)
        USING embedding::vector(1536);

ALTER TABLE langchain_pg_embedding
    ADD COLUMN if not exists created_at timestamp DEFAULT now();

ALTER TABLE langchain_pg_embedding
    ADD COLUMN if not exists modified_at timestamp DEFAULT now();

ALTER TABLE langchain_pg_embedding
    ADD COLUMN if not exists created_by varchar(255) DEfAULT 'system';

ALTER TABLE langchain_pg_embedding
    ADD COLUMN if not exists modified_by varchar(255) DEfAULT 'system';

create index if not exists ix_langchain_pg_embedding_embedding
    on langchain_pg_embedding using hnsw (embedding vector_cosine_ops);


