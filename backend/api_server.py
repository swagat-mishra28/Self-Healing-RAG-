"""
FastAPI Server for Self-Healing RAG System

Provides REST API and WebSocket endpoints for the RAG system
"""
import time
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Database imports
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Load environment variables
load_dotenv()

# Import RAG system
from self_healing_rag import SelfHealingRAGSystem
from reranker import BiEncoderVsCrossEncoderDemo

# Import database models and session
try:
    from backend.database import (
        get_session, 
        QueryMetric, 
        FeedbackExample, 
        init_db,
        AsyncSessionLocal
    )
except ImportError:
    print("⚠️  Warning: Could not import database module. Database logging disabled.")
    get_session = None
    QueryMetric = None
    FeedbackExample = None
    init_db = None
    AsyncSessionLocal = None


# ============================================================================
# APP INITIALIZATION WITH LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    print("🚀 Starting Self-Healing RAG API Server...")
    
    global rag_system
    
    try:
        # Initialize database
        if init_db is not None:
            await init_db()
            print("✅ Database initialized")
        
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("⚠️  WARNING: OPENAI_API_KEY not found in environment")
        
        # Initialize RAG system
        rag_system = SelfHealingRAGSystem(
            openai_api_key=openai_api_key,
            enable_web_search=False
        )
        
        # Load sample documents
        rag_system.load_sample_documents()
        
        print("✅ RAG System initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        traceback.print_exc()
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Self-Healing RAG",
    description="Production RAG with self-critique and feedback loops",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[SelfHealingRAGSystem] = None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    enable_hyde: bool = True
    enable_decomposition: bool = True
    enable_crag: bool = True
    enable_reranking: bool = True
    enable_learning: bool = True


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    is_positive: bool


class DocumentUpload(BaseModel):
    documents: List[str]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "service": "Self-Healing RAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "feedback": "/api/feedback",
            "statistics": "/api/statistics",
            "upload": "/api/upload",
            "architecture": "/api/architecture",
            "websocket": "/ws"
        }
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "system_ready": rag_system.vector_index is not None,
        "components": {
            "query_decomposer": "ready",
            "hyde_engine": "ready" if rag_system.hyde_engine else "not_loaded",
            "crag_system": "ready",
            "reranker": "ready",
            "learning_manager": "ready"
        }
    }


@app.post("/api/query")
async def query_rag(
    request: QueryRequest, 
    session: AsyncSession = Depends(get_session) if get_session else None
):
    """
    Process a query through the RAG pipeline
    
    Args:
        request: Query request with configuration
        session: Database session (optional)
        
    Returns:
        Query result with answer and metadata
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.vector_index:
        raise HTTPException(status_code=400, detail="No documents loaded")
    
    start = time.time()
    
    try:
        # Process query through RAG pipeline
        result = rag_system.process_query(
            query=request.query,
            enable_decomposition=request.enable_decomposition,
            enable_hyde=request.enable_hyde,
            enable_crag=request.enable_crag,
            enable_reranking=request.enable_reranking,
            enable_learning=request.enable_learning
        )
        
        # Log to database if available
        if session is not None and QueryMetric is not None:
            try:
                execution_time_ms = (time.time() - start) * 1000
                metric = QueryMetric(
                    query=request.query,
                    techniques_used=["hyde", "decomposition", "crag", "reranking"],
                    execution_time_ms=execution_time_ms,
                    documents_retrieved=len(result.get("all_docs", [])),
                    final_documents_used=len(result.get("final_docs", [])),
                    confidence_score=result.get("confidence", 0.0),
                    status="success"
                )
                session.add(metric)
                await session.commit()
            except Exception as db_error:
                print(f"Warning: Could not log query to database: {db_error}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        
        # Log error to database if available
        if session is not None and QueryMetric is not None:
            try:
                execution_time_ms = (time.time() - start) * 1000
                metric = QueryMetric(
                    query=request.query,
                    execution_time_ms=execution_time_ms,
                    status="error",
                    error_message=str(e)
                )
                session.add(metric)
                await session.commit()
            except Exception as db_error:
                print(f"Warning: Could not log error to database: {db_error}")
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    session: AsyncSession = Depends(get_session) if get_session else None
):
    """
    Submit user feedback for learning
    
    Args:
        request: Feedback data
        session: Database session (optional)
        
    Returns:
        Confirmation
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Add feedback to RAG system learning
        rag_system.add_feedback(
            query=request.query,
            answer=request.answer,
            is_positive=request.is_positive
        )
        
        # Store in database if available
        feedback_id = None
        if session is not None and FeedbackExample is not None:
            try:
                example = FeedbackExample(
                    query=request.query,
                    answer=request.answer,
                    is_positive=request.is_positive,
                    feedback_type="user_feedback"
                )
                session.add(example)
                await session.commit()
                feedback_id = example.id
            except Exception as db_error:
                print(f"Warning: Could not store feedback in database: {db_error}")
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "feedback_id": feedback_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics(
    session: AsyncSession = Depends(get_session) if get_session else None
):
    """
    Get system performance statistics
    
    Args:
        session: Database session (optional)
        
    Returns:
        Statistics dictionary
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get RAG system stats
        stats = rag_system.get_statistics()
        learning_stats = rag_system.learning_manager.get_example_stats()
        
        db_stats = {}
        
        # Get database stats if available
        if session is not None and QueryMetric is not None:
            try:
                total_queries_result = await session.execute(
                    select(func.count()).select_from(QueryMetric)
                )
                total_queries = total_queries_result.scalar() or 0
                
                avg_time_result = await session.execute(
                    select(func.avg(QueryMetric.execution_time_ms)).select_from(QueryMetric)
                )
                avg_execution_time = avg_time_result.scalar() or 0
                
                success_count_result = await session.execute(
                    select(func.count()).select_from(QueryMetric).where(
                        QueryMetric.status == "success"
                    )
                )
                success_count = success_count_result.scalar() or 0
                
                yesterday = datetime.utcnow() - timedelta(days=1)
                last_24h_result = await session.execute(
                    select(func.count()).select_from(QueryMetric).where(
                        QueryMetric.created_at >= yesterday
                    )
                )
                last_24h = last_24h_result.scalar() or 0
                
                db_stats = {
                    "database_stats": {
                        "total_queries_logged": total_queries,
                        "queries_last_24h": last_24h,
                        "success_rate": round(
                            (success_count / max(total_queries, 1)) * 100, 2
                        ),
                        "avg_execution_time_ms": round(avg_execution_time, 2)
                    }
                }
                
                if FeedbackExample is not None:
                    total_feedback_result = await session.execute(
                        select(func.count()).select_from(FeedbackExample)
                    )
                    total_feedback = total_feedback_result.scalar() or 0
                    
                    positive_feedback_result = await session.execute(
                        select(func.count()).select_from(FeedbackExample).where(
                            FeedbackExample.is_positive == True
                        )
                    )
                    positive_feedback = positive_feedback_result.scalar() or 0
                    
                    db_stats["feedback_stats"] = {
                        "total_feedback": total_feedback,
                        "positive_feedback": positive_feedback,
                        "feedback_ratio": round(
                            (positive_feedback / max(total_feedback, 1)) * 100, 2
                        )
                    }
                    
            except Exception as db_error:
                print(f"Warning: Could not retrieve database statistics: {db_error}")
        
        return {
            "system_stats": stats,
            "learning_stats": learning_stats,
            **db_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_documents(request: DocumentUpload):
    """
    Upload custom documents to the RAG system
    
    Args:
        request: Document upload request
        
    Returns:
        Confirmation
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        rag_system.load_documents(request.documents)
        
        return {
            "status": "success",
            "message": f"Loaded {len(request.documents)} documents",
            "document_count": len(request.documents)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/architecture")
async def get_architecture_info():
    """
    Get architecture information and comparisons
    
    Returns:
        Architecture details
    """
    try:
        bi_vs_cross = BiEncoderVsCrossEncoderDemo.explain_difference()
        
        return {
            "encoder_comparison": bi_vs_cross,
            "pipeline_stages": {
                "1_query_enhancement": {
                    "techniques": ["HyDE", "Query Decomposition"],
                    "purpose": "Transform raw queries into optimal retrieval requests"
                },
                "2_retrieval": {
                    "techniques": ["Vector Search", "Bi-Encoder"],
                    "purpose": "Fast semantic recall of candidate documents"
                },
                "3_validation": {
                    "techniques": ["CRAG", "Relevance Grading"],
                    "purpose": "Filter irrelevant documents, trigger fallback"
                },
                "4_reranking": {
                    "techniques": ["Cross-Encoder"],
                    "purpose": "Precision scoring for final document selection"
                },
                "5_generation": {
                    "techniques": ["Dynamic Few-Shot", "LLM Generation"],
                    "purpose": "Generate accurate answer from validated context"
                },
                "6_learning": {
                    "techniques": ["Feedback Loop", "Example Storage"],
                    "purpose": "Continuous improvement from successful interactions"
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time query processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query results
    
    Clients can send queries and receive real-time updates
    """
    await websocket.accept()
    
    if rag_system is None:
        await websocket.send_json({
            "type": "error",
            "message": "RAG system not initialized"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            query = data.get("query", "")
            config = data.get("config", {})
            
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty query"
                })
                continue
            
            # Send processing started event
            await websocket.send_json({
                "type": "processing_started",
                "query": query
            })
            
            try:
                # Process query
                result = rag_system.process_query(
                    query=query,
                    enable_decomposition=config.get("enable_decomposition", True),
                    enable_hyde=config.get("enable_hyde", True),
                    enable_crag=config.get("enable_crag", True),
                    enable_reranking=config.get("enable_reranking", True),
                    enable_learning=config.get("enable_learning", True)
                )
                
                # Send result
                await websocket.send_json({
                    "type": "result",
                    "data": result
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🌟 Starting Self-Healing RAG API Server...")
    print("📚 Loading environment variables...")
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
