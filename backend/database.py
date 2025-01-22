from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, select, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import json

# Create async engine
DATABASE_URL = "sqlite+aiosqlite:///./chats.db"
engine = create_async_engine(DATABASE_URL, echo=True)

# Create base class for declarative models
class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    audio_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# Create async session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

async def create_conversation():
    """Create a new conversation"""
    async with async_session() as session:
        conversation = Conversation()
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation

async def add_message(
    conversation_id: int,
    role: str,
    content: str,
    audio_path: str = None
):
    """Add a message to a conversation"""
    async with async_session() as session:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            audio_path=audio_path
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message

async def get_conversation(conversation_id: int):
    """Get a conversation by ID with all its messages"""
    async with async_session() as session:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt)
        conversation = result.scalar_one_or_none()
        
        if conversation:
            # Load messages relationship
            await session.refresh(conversation, attribute_names=['messages'])
            return {
                'id': conversation.id,
                'started_at': conversation.started_at.isoformat(),
                'messages': [
                    {
                        'id': msg.id,
                        'role': msg.role,
                        'content': msg.content,
                        'audio_path': msg.audio_path,
                        'created_at': msg.created_at.isoformat()
                    }
                    for msg in conversation.messages
                ]
            }
        return None

async def list_conversations(skip: int = 0, limit: int = 10):
    """List conversations with pagination"""
    async with async_session() as session:
        # Use select statement instead of raw SQL
        stmt = select(Conversation).order_by(Conversation.started_at.desc()).offset(skip).limit(limit)
        result = await session.execute(stmt)
        conversations = result.scalars().all()
        
        return [
            {
                'id': conv.id,
                'started_at': conv.started_at.isoformat()
            }
            for conv in conversations
        ]

async def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages"""
    async with async_session() as session:
        # Delete messages first
        stmt_messages = select(Message).where(Message.conversation_id == conversation_id)
        result = await session.execute(stmt_messages)
        messages = result.scalars().all()
        for message in messages:
            await session.delete(message)
        
        # Then delete conversation
        stmt_conv = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt_conv)
        conversation = result.scalar_one_or_none()
        if conversation:
            await session.delete(conversation)
        
        await session.commit()