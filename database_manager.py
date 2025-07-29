#!/usr/bin/env python3
"""
PostgreSQL Database Manager for Announcement Context Storage
Handles persistent storage of announcement context and history
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pg8000

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

# Database configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'username': os.getenv('DB_USERNAME'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', '5433'))
}

# Validate that all required environment variables are set
required_vars = ['DB_HOST', 'DB_USERNAME', 'DB_PASSWORD', 'DB_NAME']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

Base = declarative_base()

class AnnouncementContext(Base):
    """SQLAlchemy model for announcement context"""
    __tablename__ = 'announcement_context'
    
    id = Column(Integer, primary_key=True)
    announcement_type = Column(String(100), nullable=False)
    original_title = Column(String(500), nullable=False)
    original_message = Column(Text, nullable=False)
    enhanced_title = Column(String(500), nullable=False)
    enhanced_message = Column(Text, nullable=False)
    target_audience = Column(String(100), nullable=False)
    tone = Column(String(50), nullable=False)
    context_connections = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class TitleEnhancementContext(Base):
    """SQLAlchemy model for title enhancement context"""
    __tablename__ = 'title_enhancement_context'
    
    id = Column(Integer, primary_key=True)
    original_title = Column(String(500), nullable=False)
    enhanced_titles = Column(Text, nullable=False)  # JSON array of enhanced titles
    context_type = Column(String(100), nullable=False)
    domain = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class MessageEnhancementContext(Base):
    """SQLAlchemy model for message enhancement context"""
    __tablename__ = 'message_enhancement_context'
    
    id = Column(Integer, primary_key=True)
    original_message = Column(Text, nullable=False)
    enhanced_messages = Column(Text, nullable=False)  # JSON array of enhanced messages
    context_type = Column(String(100), nullable=False)
    domain = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class LLMUsageDaily(Base):
    """SQLAlchemy model for daily LLM usage tracking"""
    __tablename__ = 'llm_usage_daily'
    
    id = Column(Integer, primary_key=True)
    date = Column(String(10), nullable=False, unique=True)  # YYYY-MM-DD format
    text_requests = Column(Integer, default=0)
    image_requests = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    requests_by_type = Column(Text)  # JSON string of request types
    peak_hour = Column(String(5))  # HH:MM format
    requests_timeline = Column(Text)  # JSON array of request timeline
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LLMUsageMonthly(Base):
    """SQLAlchemy model for monthly LLM usage tracking"""
    __tablename__ = 'llm_usage_monthly'
    
    id = Column(Integer, primary_key=True)
    month = Column(String(7), nullable=False, unique=True)  # YYYY-MM format
    text_requests = Column(Integer, default=0)
    image_requests = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    avg_daily_requests = Column(Float, default=0.0)
    peak_day = Column(String(10))  # YYYY-MM-DD format
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LLMUsageTotal(Base):
    """SQLAlchemy model for total LLM usage statistics"""
    __tablename__ = 'llm_usage_total'
    
    id = Column(Integer, primary_key=True)
    total_text_requests = Column(Integer, default=0)
    total_image_requests = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    first_usage_date = Column(String(10))  # YYYY-MM-DD format
    last_usage_date = Column(String(10))  # YYYY-MM-DD format
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LLMUsageTrends(Base):
    """SQLAlchemy model for LLM usage trends and analytics"""
    __tablename__ = 'llm_usage_trends'
    
    id = Column(Integer, primary_key=True)
    trend_type = Column(String(50), nullable=False)  # 'daily_averages', 'peak_usage_days', 'cost_trends'
    trend_data = Column(Text, nullable=False)  # JSON string of trend data
    analysis_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DatabaseManager:
    """Manage PostgreSQL database operations for announcement context"""
    
    def __init__(self):
        """Initialize database connection"""
        try:
            # Create connection string for pg8000
            connection_string = f"postgresql+pg8000://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            
            # Create engine with pg8000
            self.engine = create_engine(
                connection_string,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create session factory
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.SessionLocal = SessionLocal
            
            # Initialize database
            self.init_database()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    def init_database(self):
        """Initialize database tables"""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def add_announcement_context(self, announcement_type: str, original_title: str, 
                                original_message: str, enhanced_title: str, 
                                enhanced_message: str, target_audience: str, 
                                tone: str, context_connections: str = None) -> bool:
        """Add announcement to context database"""
        try:
            session = self.SessionLocal()
            
            announcement = AnnouncementContext(
                announcement_type=announcement_type,
                original_title=original_title,
                original_message=original_message,
                enhanced_title=enhanced_title,
                enhanced_message=enhanced_message,
                target_audience=target_audience,
                tone=tone,
                context_connections=context_connections
            )
            
            session.add(announcement)
            session.commit()
            session.close()
            
            logger.info(f"Added announcement to context: {enhanced_title[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add announcement to context: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_recent_announcements(self, limit: int = 5) -> List[Dict]:
        """Get recent announcements from database"""
        try:
            session = self.SessionLocal()
            
            announcements = session.query(AnnouncementContext)\
                .filter(AnnouncementContext.is_active == True)\
                .order_by(AnnouncementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            result = []
            for announcement in announcements:
                result.append({
                    'id': announcement.id,
                    'announcement_type': announcement.announcement_type,
                    'original_title': announcement.original_title,
                    'enhanced_title': announcement.enhanced_title,
                    'enhanced_message': announcement.enhanced_message,
                    'target_audience': announcement.target_audience,
                    'tone': announcement.tone,
                    'context_connections': announcement.context_connections,
                    'created_at': announcement.created_at.isoformat() if announcement.created_at else None
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get recent announcements: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_announcements_by_type(self, announcement_type: str, limit: int = 5) -> List[Dict]:
        """Get announcements by type"""
        try:
            session = self.SessionLocal()
            
            announcements = session.query(AnnouncementContext)\
                .filter(AnnouncementContext.announcement_type == announcement_type)\
                .filter(AnnouncementContext.is_active == True)\
                .order_by(AnnouncementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            result = []
            for announcement in announcements:
                result.append({
                    'id': announcement.id,
                    'announcement_type': announcement.announcement_type,
                    'original_title': announcement.original_title,
                    'enhanced_title': announcement.enhanced_title,
                    'enhanced_message': announcement.enhanced_message,
                    'target_audience': announcement.target_audience,
                    'tone': announcement.tone,
                    'context_connections': announcement.context_connections,
                    'created_at': announcement.created_at.isoformat() if announcement.created_at else None
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get announcements by type: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_announcements_by_audience(self, target_audience: str, limit: int = 5) -> List[Dict]:
        """Get announcements by target audience"""
        try:
            session = self.SessionLocal()
            
            announcements = session.query(AnnouncementContext)\
                .filter(AnnouncementContext.target_audience == target_audience)\
                .filter(AnnouncementContext.is_active == True)\
                .order_by(AnnouncementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            result = []
            for announcement in announcements:
                result.append({
                    'id': announcement.id,
                    'announcement_type': announcement.announcement_type,
                    'original_title': announcement.original_title,
                    'enhanced_title': announcement.enhanced_title,
                    'enhanced_message': announcement.enhanced_message,
                    'target_audience': announcement.target_audience,
                    'tone': announcement.tone,
                    'context_connections': announcement.context_connections,
                    'created_at': announcement.created_at.isoformat() if announcement.created_at else None
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get announcements by audience: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_context_summary(self, limit: int = 3) -> str:
        """Get a summary of recent announcements for AI context"""
        try:
            recent_announcements = self.get_recent_announcements(limit)
            
            if not recent_announcements:
                return "No previous announcements."
            
            summary_parts = []
            for announcement in recent_announcements:
                summary_parts.append(
                    f"Type: {announcement['announcement_type']}, "
                    f"Audience: {announcement['target_audience']}, "
                    f"Title: {announcement['enhanced_title']}, "
                    f"Tone: {announcement['tone']}"
                )
            
            return "Recent announcements: " + "; ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to get context summary: {e}")
            return "No previous announcements."
    
    def get_announcement_stats(self) -> Dict[str, Any]:
        """Get announcement statistics"""
        try:
            session = self.SessionLocal()
            
            # Total announcements
            total_count = session.query(AnnouncementContext)\
                .filter(AnnouncementContext.is_active == True)\
                .count()
            
            # Count by type
            type_counts = {}
            types = session.query(AnnouncementContext.announcement_type)\
                .filter(AnnouncementContext.is_active == True)\
                .distinct()\
                .all()
            
            for type_tuple in types:
                announcement_type = type_tuple[0]
                count = session.query(AnnouncementContext)\
                    .filter(AnnouncementContext.announcement_type == announcement_type)\
                    .filter(AnnouncementContext.is_active == True)\
                    .count()
                type_counts[announcement_type] = count
            
            # Count by audience
            audience_counts = {}
            audiences = session.query(AnnouncementContext.target_audience)\
                .filter(AnnouncementContext.is_active == True)\
                .distinct()\
                .all()
            
            for audience_tuple in audiences:
                audience = audience_tuple[0]
                count = session.query(AnnouncementContext)\
                    .filter(AnnouncementContext.target_audience == audience)\
                    .filter(AnnouncementContext.is_active == True)\
                    .count()
                audience_counts[audience] = count
            
            session.close()
            
            return {
                "total_announcements": total_count,
                "by_type": type_counts,
                "by_audience": audience_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get announcement stats: {e}")
            if 'session' in locals():
                session.close()
            return {
                "total_announcements": 0,
                "by_type": {},
                "by_audience": {}
            }
    
    def clear_context(self) -> bool:
        """Clear all announcement context (soft delete)"""
        try:
            session = self.SessionLocal()
            
            # Soft delete by setting is_active to False
            session.query(AnnouncementContext)\
                .filter(AnnouncementContext.is_active == True)\
                .update({"is_active": False})
            
            session.commit()
            session.close()
            
            logger.info("Context history cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear context: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def delete_announcement(self, announcement_id: int) -> bool:
        """Delete specific announcement (soft delete)"""
        try:
            session = self.SessionLocal()
            
            announcement = session.query(AnnouncementContext)\
                .filter(AnnouncementContext.id == announcement_id)\
                .filter(AnnouncementContext.is_active == True)\
                .first()
            
            if announcement:
                announcement.is_active = False
                session.commit()
                session.close()
                logger.info(f"Announcement {announcement_id} deleted successfully")
                return True
            else:
                session.close()
                logger.warning(f"Announcement {announcement_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete announcement {announcement_id}: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False

    # Title Enhancement Methods
    
    def add_title_context(self, original_title: str, enhanced_titles: List[str], 
                         context_type: str = "title_enhancement", 
                         domain: str = "general") -> bool:
        """Add title enhancement context to database"""
        try:
            session = self.SessionLocal()
            
            # Convert enhanced_titles list to JSON string
            enhanced_titles_json = json.dumps(enhanced_titles)
            
            # Create new title enhancement context
            title_context = TitleEnhancementContext(
                original_title=original_title,
                enhanced_titles=enhanced_titles_json,
                context_type=context_type,
                domain=domain
            )
            
            session.add(title_context)
            session.commit()
            session.close()
            
            logger.info(f"Title enhancement context added successfully for: {original_title[:30]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add title enhancement context: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_title_context_summary(self, domain: str = "general", limit: int = 3) -> str:
        """Get a summary of recent title enhancements for context"""
        try:
            session = self.SessionLocal()
            
            # Get recent title enhancements for the domain
            recent_enhancements = session.query(TitleEnhancementContext)\
                .filter(TitleEnhancementContext.domain == domain)\
                .filter(TitleEnhancementContext.is_active == True)\
                .order_by(TitleEnhancementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            if not recent_enhancements:
                session.close()
                return "No previous title enhancements for this domain."
            
            # Build context summary
            summary_parts = []
            for enhancement in recent_enhancements:
                try:
                    enhanced_titles = json.loads(enhancement.enhanced_titles)
                    titles_str = ", ".join(enhanced_titles[:2])  # Show first 2 titles
                    summary_parts.append(f"Original: '{enhancement.original_title}' → Enhanced: {titles_str}")
                except json.JSONDecodeError:
                    summary_parts.append(f"Original: '{enhancement.original_title}' → Enhanced titles")
            
            session.close()
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to get title context summary: {e}")
            if 'session' in locals():
                session.close()
            return "No previous title enhancements."
    
    def get_recent_title_enhancements(self, domain: str = "general", limit: int = 5) -> List[Dict]:
        """Get recent title enhancements from database"""
        try:
            session = self.SessionLocal()
            
            recent_enhancements = session.query(TitleEnhancementContext)\
                .filter(TitleEnhancementContext.domain == domain)\
                .filter(TitleEnhancementContext.is_active == True)\
                .order_by(TitleEnhancementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            enhancements = []
            for enhancement in recent_enhancements:
                try:
                    enhanced_titles = json.loads(enhancement.enhanced_titles)
                except json.JSONDecodeError:
                    enhanced_titles = []
                
                enhancements.append({
                    "id": enhancement.id,
                    "original_title": enhancement.original_title,
                    "enhanced_titles": enhanced_titles,
                    "context_type": enhancement.context_type,
                    "domain": enhancement.domain,
                    "created_at": enhancement.created_at.isoformat()
                })
            
            session.close()
            return enhancements
            
        except Exception as e:
            logger.error(f"Failed to get recent title enhancements: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_title_enhancements_by_type(self, context_type: str, domain: str = "general", limit: int = 5) -> List[Dict]:
        """Get title enhancements by context type"""
        try:
            session = self.SessionLocal()
            
            enhancements = session.query(TitleEnhancementContext)\
                .filter(TitleEnhancementContext.context_type == context_type)\
                .filter(TitleEnhancementContext.domain == domain)\
                .filter(TitleEnhancementContext.is_active == True)\
                .order_by(TitleEnhancementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            result = []
            for enhancement in enhancements:
                try:
                    enhanced_titles = json.loads(enhancement.enhanced_titles)
                except json.JSONDecodeError:
                    enhanced_titles = []
                
                result.append({
                    "id": enhancement.id,
                    "original_title": enhancement.original_title,
                    "enhanced_titles": enhanced_titles,
                    "context_type": enhancement.context_type,
                    "domain": enhancement.domain,
                    "created_at": enhancement.created_at.isoformat()
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get title enhancements by type: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_title_enhancement_stats(self, domain: str = "general") -> Dict[str, Any]:
        """Get title enhancement statistics"""
        try:
            session = self.SessionLocal()
            
            # Total count
            total_count = session.query(TitleEnhancementContext)\
                .filter(TitleEnhancementContext.domain == domain)\
                .filter(TitleEnhancementContext.is_active == True)\
                .count()
            
            # Count by context type
            type_counts = {}
            types = session.query(TitleEnhancementContext.context_type)\
                .filter(TitleEnhancementContext.domain == domain)\
                .filter(TitleEnhancementContext.is_active == True)\
                .distinct()\
                .all()
            
            for type_tuple in types:
                context_type = type_tuple[0]
                count = session.query(TitleEnhancementContext)\
                    .filter(TitleEnhancementContext.context_type == context_type)\
                    .filter(TitleEnhancementContext.domain == domain)\
                    .filter(TitleEnhancementContext.is_active == True)\
                    .count()
                type_counts[context_type] = count
            
            session.close()
            
            return {
                "total_enhancements": total_count,
                "by_context_type": type_counts,
                "domain": domain
            }
            
        except Exception as e:
            logger.error(f"Failed to get title enhancement stats: {e}")
            if 'session' in locals():
                session.close()
            return {
                "total_enhancements": 0,
                "by_context_type": {},
                "domain": domain
            }
    
    def clear_title_context(self, domain: str = None) -> bool:
        """Clear title enhancement context (soft delete)"""
        try:
            session = self.SessionLocal()
            
            if domain:
                # Clear context for specific domain
                session.query(TitleEnhancementContext)\
                    .filter(TitleEnhancementContext.domain == domain)\
                    .filter(TitleEnhancementContext.is_active == True)\
                    .update({"is_active": False})
            else:
                # Clear all title enhancement context
                session.query(TitleEnhancementContext)\
                    .filter(TitleEnhancementContext.is_active == True)\
                    .update({"is_active": False})
            
            session.commit()
            session.close()
            
            logger.info(f"Title enhancement context cleared successfully for domain: {domain or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear title enhancement context: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def delete_title_enhancement(self, enhancement_id: int) -> bool:
        """Delete specific title enhancement (soft delete)"""
        try:
            session = self.SessionLocal()
            
            enhancement = session.query(TitleEnhancementContext)\
                .filter(TitleEnhancementContext.id == enhancement_id)\
                .filter(TitleEnhancementContext.is_active == True)\
                .first()
            
            if enhancement:
                enhancement.is_active = False
                session.commit()
                session.close()
                logger.info(f"Title enhancement {enhancement_id} deleted successfully")
                return True
            else:
                session.close()
                logger.warning(f"Title enhancement {enhancement_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete title enhancement {enhancement_id}: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False

    # Message Enhancement Context Methods
    
    def add_message_context(self, original_message: str, enhanced_messages: List[str], 
                          context_type: str = "message_enhancement", 
                          domain: str = "general") -> bool:
        """Add message enhancement to database context"""
        try:
            session = self.SessionLocal()
            
            # Convert enhanced messages list to JSON string
            enhanced_messages_json = json.dumps(enhanced_messages)
            
            # Create new message enhancement context
            message_context = MessageEnhancementContext(
                original_message=original_message,
                enhanced_messages=enhanced_messages_json,
                context_type=context_type,
                domain=domain
            )
            
            session.add(message_context)
            session.commit()
            session.close()
            
            logger.info(f"Message enhancement context added successfully for domain: {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message context: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_message_context_summary(self, domain: str = "general", limit: int = 3) -> str:
        """Get a summary of recent message enhancements for context"""
        try:
            session = self.SessionLocal()
            
            recent_messages = session.query(MessageEnhancementContext)\
                .filter(MessageEnhancementContext.domain == domain)\
                .filter(MessageEnhancementContext.is_active == True)\
                .order_by(MessageEnhancementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            session.close()
            
            if not recent_messages:
                return "No previous message enhancements for this domain."
            
            summary_parts = []
            for msg in recent_messages:
                try:
                    enhanced_messages = json.loads(msg.enhanced_messages)
                    summary_parts.append(f"Original: '{msg.original_message[:50]}...' → Enhanced: {', '.join(enhanced_messages[:2])}")
                except json.JSONDecodeError:
                    summary_parts.append(f"Original: '{msg.original_message[:50]}...' → Enhanced messages available")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to get message context summary: {e}")
            if 'session' in locals():
                session.close()
            return "Error retrieving message context summary."
    
    def get_recent_message_enhancements(self, domain: str = "general", limit: int = 5) -> List[Dict]:
        """Get recent message enhancements from database"""
        try:
            session = self.SessionLocal()
            
            recent_messages = session.query(MessageEnhancementContext)\
                .filter(MessageEnhancementContext.domain == domain)\
                .filter(MessageEnhancementContext.is_active == True)\
                .order_by(MessageEnhancementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            session.close()
            
            enhancements = []
            for msg in recent_messages:
                try:
                    enhanced_messages = json.loads(msg.enhanced_messages)
                    enhancements.append({
                        "id": msg.id,
                        "original_message": msg.original_message,
                        "enhanced_messages": enhanced_messages,
                        "context_type": msg.context_type,
                        "domain": msg.domain,
                        "created_at": msg.created_at.isoformat()
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in message enhancement {msg.id}")
                    continue
            
            return enhancements
            
        except Exception as e:
            logger.error(f"Failed to get recent message enhancements: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_message_enhancements_by_type(self, context_type: str, domain: str = "general", limit: int = 5) -> List[Dict]:
        """Get message enhancements filtered by context type"""
        try:
            session = self.SessionLocal()
            
            messages = session.query(MessageEnhancementContext)\
                .filter(MessageEnhancementContext.context_type == context_type)\
                .filter(MessageEnhancementContext.domain == domain)\
                .filter(MessageEnhancementContext.is_active == True)\
                .order_by(MessageEnhancementContext.created_at.desc())\
                .limit(limit)\
                .all()
            
            session.close()
            
            enhancements = []
            for msg in messages:
                try:
                    enhanced_messages = json.loads(msg.enhanced_messages)
                    enhancements.append({
                        "id": msg.id,
                        "original_message": msg.original_message,
                        "enhanced_messages": enhanced_messages,
                        "context_type": msg.context_type,
                        "domain": msg.domain,
                        "created_at": msg.created_at.isoformat()
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in message enhancement {msg.id}")
                    continue
            
            return enhancements
            
        except Exception as e:
            logger.error(f"Failed to get message enhancements by type: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def get_message_enhancement_stats(self, domain: str = "general") -> Dict[str, Any]:
        """Get statistics about message enhancements"""
        try:
            session = self.SessionLocal()
            
            # Total enhancements
            total_count = session.query(MessageEnhancementContext)\
                .filter(MessageEnhancementContext.domain == domain)\
                .filter(MessageEnhancementContext.is_active == True)\
                .count()
            
            # Enhancements by context type
            context_type_counts = {}
            context_types = session.query(MessageEnhancementContext.context_type)\
                .filter(MessageEnhancementContext.domain == domain)\
                .filter(MessageEnhancementContext.is_active == True)\
                .distinct()\
                .all()
            
            for context_type_tuple in context_types:
                context_type = context_type_tuple[0]
                count = session.query(MessageEnhancementContext)\
                    .filter(MessageEnhancementContext.context_type == context_type)\
                    .filter(MessageEnhancementContext.domain == domain)\
                    .filter(MessageEnhancementContext.is_active == True)\
                    .count()
                context_type_counts[context_type] = count
            
            # Recent activity (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_activity = session.query(MessageEnhancementContext)\
                .filter(MessageEnhancementContext.domain == domain)\
                .filter(MessageEnhancementContext.is_active == True)\
                .filter(MessageEnhancementContext.created_at >= week_ago)\
                .count()
            
            session.close()
            
            return {
                "total_enhancements": total_count,
                "by_context_type": context_type_counts,
                "domain": domain,
                "recent_activity_7_days": recent_activity
            }
            
        except Exception as e:
            logger.error(f"Failed to get message enhancement stats: {e}")
            if 'session' in locals():
                session.close()
            return {
                "total_enhancements": 0,
                "by_context_type": {},
                "domain": domain,
                "recent_activity_7_days": 0
            }
    
    def clear_message_context(self, domain: str = None) -> bool:
        """Clear message enhancement context history (soft delete)"""
        try:
            session = self.SessionLocal()
            
            if domain:
                # Clear specific domain
                session.query(MessageEnhancementContext)\
                    .filter(MessageEnhancementContext.domain == domain)\
                    .filter(MessageEnhancementContext.is_active == True)\
                    .update({"is_active": False})
                logger.info(f"Message context cleared for domain: {domain}")
            else:
                # Clear all domains
                session.query(MessageEnhancementContext)\
                    .filter(MessageEnhancementContext.is_active == True)\
                    .update({"is_active": False})
                logger.info("All message context cleared")
            
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear message context: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def delete_message_enhancement(self, enhancement_id: int) -> bool:
        """Soft delete a specific message enhancement"""
        try:
            session = self.SessionLocal()
            
            enhancement = session.query(MessageEnhancementContext)\
                .filter(MessageEnhancementContext.id == enhancement_id)\
                .filter(MessageEnhancementContext.is_active == True)\
                .first()
            
            if enhancement:
                enhancement.is_active = False
                session.commit()
                session.close()
                logger.info(f"Message enhancement {enhancement_id} deleted successfully")
                return True
            else:
                session.close()
                logger.warning(f"Message enhancement {enhancement_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete message enhancement {enhancement_id}: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False

    # LLM Usage Tracking Methods
    
    def add_llm_text_usage(self, tokens_used: int, cost_usd: float, request_type: str = "text_generation") -> bool:
        """Add text generation usage to database"""
        try:
            session = self.SessionLocal()
            today = datetime.now().strftime('%Y-%m-%d')
            month = datetime.now().strftime('%Y-%m')
            now = datetime.now().isoformat()
            
            # Update or create daily usage
            daily_usage = session.query(LLMUsageDaily)\
                .filter(LLMUsageDaily.date == today)\
                .first()
            
            if daily_usage:
                daily_usage.text_requests += 1
                daily_usage.tokens_used += tokens_used
                daily_usage.cost_usd += cost_usd
                daily_usage.updated_at = datetime.utcnow()
                
                # Update requests by type
                try:
                    requests_by_type = json.loads(daily_usage.requests_by_type) if daily_usage.requests_by_type else {}
                except:
                    requests_by_type = {}
                
                requests_by_type[request_type] = requests_by_type.get(request_type, 0) + 1
                daily_usage.requests_by_type = json.dumps(requests_by_type)
                
                # Update timeline
                try:
                    timeline = json.loads(daily_usage.requests_timeline) if daily_usage.requests_timeline else []
                except:
                    timeline = []
                
                timeline.append({
                    "timestamp": now,
                    "type": "text",
                    "tokens": tokens_used,
                    "cost": cost_usd,
                    "request_type": request_type
                })
                daily_usage.requests_timeline = json.dumps(timeline)
                
            else:
                # Create new daily usage record
                daily_usage = LLMUsageDaily(
                    date=today,
                    text_requests=1,
                    image_requests=0,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    requests_by_type=json.dumps({request_type: 1}),
                    requests_timeline=json.dumps([{
                        "timestamp": now,
                        "type": "text",
                        "tokens": tokens_used,
                        "cost": cost_usd,
                        "request_type": request_type
                    }])
                )
                session.add(daily_usage)
            
            # Update or create monthly usage
            monthly_usage = session.query(LLMUsageMonthly)\
                .filter(LLMUsageMonthly.month == month)\
                .first()
            
            if monthly_usage:
                monthly_usage.text_requests += 1
                monthly_usage.tokens_used += tokens_used
                monthly_usage.cost_usd += cost_usd
                monthly_usage.updated_at = datetime.utcnow()
            else:
                monthly_usage = LLMUsageMonthly(
                    month=month,
                    text_requests=1,
                    image_requests=0,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd
                )
                session.add(monthly_usage)
            
            # Update total stats
            total_stats = session.query(LLMUsageTotal).first()
            if total_stats:
                total_stats.total_text_requests += 1
                total_stats.total_tokens_used += tokens_used
                total_stats.total_cost_usd += cost_usd
                total_stats.last_usage_date = today
                total_stats.updated_at = datetime.utcnow()
                
                if not total_stats.first_usage_date:
                    total_stats.first_usage_date = today
            else:
                total_stats = LLMUsageTotal(
                    total_text_requests=1,
                    total_image_requests=0,
                    total_tokens_used=tokens_used,
                    total_cost_usd=cost_usd,
                    first_usage_date=today,
                    last_usage_date=today
                )
                session.add(total_stats)
            
            session.commit()
            session.close()
            
            logger.info(f"Added LLM text usage: {tokens_used} tokens, ${cost_usd:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add LLM text usage: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def add_llm_image_usage(self, cost_usd: float, request_type: str = "image_generation") -> bool:
        """Add image generation usage to database"""
        try:
            session = self.SessionLocal()
            today = datetime.now().strftime('%Y-%m-%d')
            month = datetime.now().strftime('%Y-%m')
            now = datetime.now().isoformat()
            
            # Update or create daily usage
            daily_usage = session.query(LLMUsageDaily)\
                .filter(LLMUsageDaily.date == today)\
                .first()
            
            if daily_usage:
                daily_usage.image_requests += 1
                daily_usage.cost_usd += cost_usd
                daily_usage.updated_at = datetime.utcnow()
                
                # Update requests by type
                try:
                    requests_by_type = json.loads(daily_usage.requests_by_type) if daily_usage.requests_by_type else {}
                except:
                    requests_by_type = {}
                
                requests_by_type[request_type] = requests_by_type.get(request_type, 0) + 1
                daily_usage.requests_by_type = json.dumps(requests_by_type)
                
                # Update timeline
                try:
                    timeline = json.loads(daily_usage.requests_timeline) if daily_usage.requests_timeline else []
                except:
                    timeline = []
                
                timeline.append({
                    "timestamp": now,
                    "type": "image",
                    "tokens": 0,
                    "cost": cost_usd,
                    "request_type": request_type
                })
                daily_usage.requests_timeline = json.dumps(timeline)
                
            else:
                # Create new daily usage record
                daily_usage = LLMUsageDaily(
                    date=today,
                    text_requests=0,
                    image_requests=1,
                    tokens_used=0,
                    cost_usd=cost_usd,
                    requests_by_type=json.dumps({request_type: 1}),
                    requests_timeline=json.dumps([{
                        "timestamp": now,
                        "type": "image",
                        "tokens": 0,
                        "cost": cost_usd,
                        "request_type": request_type
                    }])
                )
                session.add(daily_usage)
            
            # Update or create monthly usage
            monthly_usage = session.query(LLMUsageMonthly)\
                .filter(LLMUsageMonthly.month == month)\
                .first()
            
            if monthly_usage:
                monthly_usage.image_requests += 1
                monthly_usage.cost_usd += cost_usd
                monthly_usage.updated_at = datetime.utcnow()
            else:
                monthly_usage = LLMUsageMonthly(
                    month=month,
                    text_requests=0,
                    image_requests=1,
                    tokens_used=0,
                    cost_usd=cost_usd
                )
                session.add(monthly_usage)
            
            # Update total stats
            total_stats = session.query(LLMUsageTotal).first()
            if total_stats:
                total_stats.total_image_requests += 1
                total_stats.total_cost_usd += cost_usd
                total_stats.last_usage_date = today
                total_stats.updated_at = datetime.utcnow()
                
                if not total_stats.first_usage_date:
                    total_stats.first_usage_date = today
            else:
                total_stats = LLMUsageTotal(
                    total_text_requests=0,
                    total_image_requests=1,
                    total_tokens_used=0,
                    total_cost_usd=cost_usd,
                    first_usage_date=today,
                    last_usage_date=today
                )
                session.add(total_stats)
            
            session.commit()
            session.close()
            
            logger.info(f"Added LLM image usage: ${cost_usd:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add LLM image usage: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_llm_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive LLM usage summary from database"""
        try:
            session = self.SessionLocal()
            today = datetime.now().strftime('%Y-%m-%d')
            month = datetime.now().strftime('%Y-%m')
            
            # Get today's usage
            daily_usage = session.query(LLMUsageDaily)\
                .filter(LLMUsageDaily.date == today)\
                .first()
            
            today_data = {
                "text_requests": daily_usage.text_requests if daily_usage else 0,
                "image_requests": daily_usage.image_requests if daily_usage else 0,
                "tokens_used": daily_usage.tokens_used if daily_usage else 0,
                "cost_usd": daily_usage.cost_usd if daily_usage else 0.0,
                "requests_by_type": {}
            }
            
            if daily_usage and daily_usage.requests_by_type:
                try:
                    today_data["requests_by_type"] = json.loads(daily_usage.requests_by_type)
                except:
                    today_data["requests_by_type"] = {}
            
            # Calculate remaining tokens (assuming 25,000 daily limit)
            daily_limit = 25000
            today_data["tokens_remaining"] = max(0, daily_limit - today_data["tokens_used"])
            
            # Get this month's usage
            monthly_usage = session.query(LLMUsageMonthly)\
                .filter(LLMUsageMonthly.month == month)\
                .first()
            
            month_data = {
                "text_requests": monthly_usage.text_requests if monthly_usage else 0,
                "image_requests": monthly_usage.image_requests if monthly_usage else 0,
                "tokens_used": monthly_usage.tokens_used if monthly_usage else 0,
                "cost_usd": monthly_usage.cost_usd if monthly_usage else 0.0
            }
            
            # Get total stats
            total_stats = session.query(LLMUsageTotal).first()
            total_data = {
                "total_text_requests": total_stats.total_text_requests if total_stats else 0,
                "total_image_requests": total_stats.total_image_requests if total_stats else 0,
                "total_tokens_used": total_stats.total_tokens_used if total_stats else 0,
                "total_cost_usd": total_stats.total_cost_usd if total_stats else 0.0,
                "first_usage_date": total_stats.first_usage_date if total_stats else None,
                "last_usage_date": total_stats.last_usage_date if total_stats else None
            }
            
            # Get usage trends
            trends_data = self._get_llm_usage_trends(session)
            
            # Calculate estimated monthly cost
            estimated_monthly_cost = self._estimate_monthly_cost(session)
            
            # Generate usage insights
            usage_insights = self._generate_llm_usage_insights(session)
            
            session.close()
            
            return {
                "current_day": {
                    "date": today,
                    **today_data
                },
                "current_month": {
                    "month": month,
                    **month_data
                },
                "total_stats": total_data,
                "usage_trends": trends_data,
                "estimated_monthly_cost": estimated_monthly_cost,
                "usage_insights": usage_insights
            }
            
        except Exception as e:
            logger.error(f"Failed to get LLM usage summary: {e}")
            if 'session' in locals():
                session.close()
            return {"error": str(e)}
    
    def get_llm_daily_usage(self, date: str) -> Dict[str, Any]:
        """Get daily LLM usage for a specific date"""
        try:
            session = self.SessionLocal()
            
            daily_usage = session.query(LLMUsageDaily)\
                .filter(LLMUsageDaily.date == date)\
                .first()
            
            if not daily_usage:
                session.close()
                return None
            
            result = {
                "text_requests": daily_usage.text_requests,
                "image_requests": daily_usage.image_requests,
                "tokens_used": daily_usage.tokens_used,
                "cost_usd": daily_usage.cost_usd,
                "peak_hour": daily_usage.peak_hour,
                "requests_by_type": {},
                "requests_timeline": []
            }
            
            if daily_usage.requests_by_type:
                try:
                    result["requests_by_type"] = json.loads(daily_usage.requests_by_type)
                except:
                    result["requests_by_type"] = {}
            
            if daily_usage.requests_timeline:
                try:
                    result["requests_timeline"] = json.loads(daily_usage.requests_timeline)
                except:
                    result["requests_timeline"] = []
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get daily LLM usage: {e}")
            if 'session' in locals():
                session.close()
            return None
    
    def get_llm_monthly_usage(self, month: str) -> Dict[str, Any]:
        """Get monthly LLM usage for a specific month"""
        try:
            session = self.SessionLocal()
            
            monthly_usage = session.query(LLMUsageMonthly)\
                .filter(LLMUsageMonthly.month == month)\
                .first()
            
            if not monthly_usage:
                session.close()
                return None
            
            result = {
                "text_requests": monthly_usage.text_requests,
                "image_requests": monthly_usage.image_requests,
                "tokens_used": monthly_usage.tokens_used,
                "cost_usd": monthly_usage.cost_usd,
                "avg_daily_requests": monthly_usage.avg_daily_requests,
                "peak_day": monthly_usage.peak_day
            }
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get monthly LLM usage: {e}")
            if 'session' in locals():
                session.close()
            return None
    
    def _get_llm_usage_trends(self, session) -> Dict[str, Any]:
        """Get LLM usage trends from database"""
        try:
            # Calculate daily averages
            daily_records = session.query(LLMUsageDaily).all()
            
            if not daily_records:
                return {
                    "daily_averages": {},
                    "peak_usage_days": [],
                    "cost_trends": []
                }
            
            total_tokens = sum(record.tokens_used for record in daily_records)
            total_cost = sum(record.cost_usd for record in daily_records)
            total_requests = sum(record.text_requests + record.image_requests for record in daily_records)
            total_days = len(daily_records)
            
            daily_averages = {
                "avg_tokens_per_day": total_tokens / total_days if total_days > 0 else 0,
                "avg_cost_per_day": total_cost / total_days if total_days > 0 else 0,
                "avg_requests_per_day": total_requests / total_days if total_days > 0 else 0
            }
            
            # Find peak usage days (top 5 by cost)
            daily_costs = [(record.date, record.cost_usd) for record in daily_records]
            daily_costs.sort(key=lambda x: x[1], reverse=True)
            peak_usage_days = daily_costs[:5]
            
            # Get cost trends (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            recent_records = [
                record for record in daily_records 
                if record.date >= thirty_days_ago
            ]
            
            cost_trends = []
            for record in recent_records:
                cost_trends.append({
                    "date": record.date,
                    "cost": record.cost_usd,
                    "tokens": record.tokens_used,
                    "requests": record.text_requests + record.image_requests
                })
            
            cost_trends.sort(key=lambda x: x["date"])
            
            return {
                "daily_averages": daily_averages,
                "peak_usage_days": peak_usage_days,
                "cost_trends": cost_trends
            }
            
        except Exception as e:
            logger.error(f"Failed to get LLM usage trends: {e}")
            return {
                "daily_averages": {},
                "peak_usage_days": [],
                "cost_trends": []
            }
    
    def _estimate_monthly_cost(self, session) -> float:
        """Estimate monthly cost based on current trends"""
        try:
            daily_records = session.query(LLMUsageDaily).all()
            
            if not daily_records:
                return 0.0
            
            total_cost = sum(record.cost_usd for record in daily_records)
            total_days = len(daily_records)
            
            if total_days > 0:
                avg_daily_cost = total_cost / total_days
                days_in_month = 30
                return avg_daily_cost * days_in_month
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to estimate monthly cost: {e}")
            return 0.0
    
    def _generate_llm_usage_insights(self, session) -> Dict[str, Any]:
        """Generate insights about LLM usage patterns"""
        try:
            insights = {
                "peak_usage_time": "Not enough data",
                "most_used_feature": "Not enough data",
                "cost_efficiency": "Good",
                "usage_growth": "Stable"
            }
            
            daily_records = session.query(LLMUsageDaily).all()
            
            if len(daily_records) < 2:
                return insights
            
            # Analyze peak usage time
            all_requests = []
            for record in daily_records:
                if record.requests_timeline:
                    try:
                        timeline = json.loads(record.requests_timeline)
                        all_requests.extend(timeline)
                    except:
                        continue
            
            if all_requests:
                # Group by hour
                hourly_counts = {}
                for request in all_requests:
                    try:
                        hour = datetime.fromisoformat(request["timestamp"]).hour
                        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
                    except:
                        continue
                
                if hourly_counts:
                    peak_hour = max(hourly_counts, key=hourly_counts.get)
                    insights["peak_usage_time"] = f"{peak_hour:02d}:00"
            
            # Analyze most used feature
            total_text = sum(record.text_requests for record in daily_records)
            total_image = sum(record.image_requests for record in daily_records)
            
            if total_text > total_image:
                insights["most_used_feature"] = "Text Generation"
            elif total_image > total_text:
                insights["most_used_feature"] = "Image Generation"
            else:
                insights["most_used_feature"] = "Balanced Usage"
            
            # Analyze cost efficiency
            total_cost = sum(record.cost_usd for record in daily_records)
            total_requests = total_text + total_image
            
            if total_requests > 0:
                cost_per_request = total_cost / total_requests
                if cost_per_request < 0.01:
                    insights["cost_efficiency"] = "Excellent"
                elif cost_per_request < 0.05:
                    insights["cost_efficiency"] = "Good"
                elif cost_per_request < 0.10:
                    insights["cost_efficiency"] = "Moderate"
                else:
                    insights["cost_efficiency"] = "High"
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate LLM usage insights: {e}")
            return {
                "peak_usage_time": "Error",
                "most_used_feature": "Error",
                "cost_efficiency": "Error",
                "usage_growth": "Error"
            }

# Create global instance
db_manager = DatabaseManager() 