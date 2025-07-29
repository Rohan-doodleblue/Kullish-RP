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
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, Boolean
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

# Create global instance
db_manager = DatabaseManager() 