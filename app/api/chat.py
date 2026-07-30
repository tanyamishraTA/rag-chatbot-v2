from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat"])

service = RAGService()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    return service.chat(
    session_id=request.session_id,
    question=request.question,
    model=request.model,
)