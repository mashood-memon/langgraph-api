from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Conversation, Message
import uuid


async def create_conversation(session: AsyncSession, user_id: str, title: str | None = None) -> str:
    conv = Conversation(user_id=user_id, title=title)
    session.add(conv)
    await session.commit()
    return str(conv.id)


async def save_message(session: AsyncSession, conversation_id: str, role: str, content: str, sources: list | None = None):
    msg = Message(conversation_id=uuid.UUID(conversation_id), role=role, content=content, sources=sources)
    session.add(msg)
    await session.commit()


async def get_recent_messages(session: AsyncSession, conversation_id: str, limit: int = 10) -> list[dict]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == uuid.UUID(conversation_id))
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


async def list_conversations(session: AsyncSession, user_id: str) -> list[dict]:
    result = await session.execute(
        select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    return [{"id": str(c.id), "title": c.title, "updated_at": c.updated_at.isoformat()} for c in convs]