from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, Index
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/self_healing_rag"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=0,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


class QueryMetric(Base):
    __tablename__ = "query_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False, index=True)
    techniques_used = Column(JSON)
    execution_time_ms = Column(Float)
    documents_retrieved = Column(Integer)
    final_documents_used = Column(Integer)
    confidence_score = Column(Float)
    status = Column(String, default="success")
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_query_created', 'query', 'created_at'),
    )


class FeedbackExample(Base):
    __tablename__ = "feedback_examples"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False, index=True)
    answer = Column(String, nullable=False)
    retrieved_docs = Column(JSON)
    is_positive = Column(Boolean, default=True, index=True)
    feedback_type = Column(String)
    user_comment = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_positive_created', 'is_positive', 'created_at'),
    )


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session