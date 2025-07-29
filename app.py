from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from openai import OpenAI
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
from dotenv import load_dotenv
from src.standard_datascience_project.utils.save_load import load_model, parse_timestamp
import pandas as pd
import numpy as np
import threading

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Swagger configuration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

# Create Swagger UI blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Automated Assessment API"
    }
)

# Register Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Import configuration
from config import Config

# Initialize OpenAI client
client = OpenAI(api_key=Config.OPENAI_API_KEY)
if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == 'your-openai-api-key-here':
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

class TokenUsageTracker:
    """Track daily token usage to enforce limits"""
    
    def __init__(self, usage_file: str):
        self.usage_file = usage_file
        self.load_usage()
    
    def load_usage(self):
        """Load token usage from file"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    self.usage_data = json.load(f)
            else:
                self.usage_data = {}
        except Exception as e:
            logger.error(f"Error loading token usage: {e}")
            self.usage_data = {}
    
    def save_usage(self):
        """Save token usage to file"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving token usage: {e}")
    
    def get_today_key(self) -> str:
        """Get today's date as string key"""
        return datetime.now().strftime('%Y-%m-%d')
    
    def check_token_limit(self, estimated_tokens: int) -> bool:
        """Check if adding tokens would exceed daily limit"""
        today = self.get_today_key()
        current_usage = self.usage_data.get(today, 0)
        return (current_usage + estimated_tokens) <= Config.MAX_TOKENS_PER_DAY
    
    def add_tokens(self, tokens_used: int):
        """Add tokens to today's usage"""
        today = self.get_today_key()
        if today not in self.usage_data:
            self.usage_data[today] = 0
        self.usage_data[today] += tokens_used
        self.save_usage()
    
    def get_daily_usage(self) -> Dict[str, Any]:
        """Get current daily usage statistics"""
        today = self.get_today_key()
        current_usage = self.usage_data.get(today, 0)
        return {
            "date": today,
            "tokens_used": current_usage,
            "tokens_remaining": Config.MAX_TOKENS_PER_DAY - current_usage,
            "limit": Config.MAX_TOKENS_PER_DAY
        }

# Initialize token tracker
token_tracker = TokenUsageTracker(Config.TOKEN_USAGE_FILE)

class AssessmentGenerator:
    """Generate assessments using OpenAI"""
    
    @staticmethod
    def generate_multiple_choice_questions(subject: str, topic: str, num_questions: int = 5) -> List[Dict]:
        """Generate multiple choice questions"""
        prompt = f"""
        Generate {num_questions} multiple choice questions for {subject} - {topic}.
        Each question should have 4 options (A, B, C, D) with only one correct answer.
        Return the response as a JSON array with the following structure:
        [
            {{
                "question": "Question text here",
                "options": {{
                    "A": "Option A",
                    "B": "Option B", 
                    "C": "Option C",
                    "D": "Option D"
                }},
                "correct_answer": "A",
                "explanation": "Brief explanation of why this is correct"
            }}
        ]
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            questions = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            raise Exception(f"Failed to generate questions: {str(e)}")
    
    @staticmethod
    def generate_essay_prompt(subject: str, topic: str) -> Dict:
        """Generate essay prompt"""
        prompt = f"""
        Create an essay prompt for {subject} - {topic}.
        Return the response as JSON with the following structure:
        {{
            "prompt": "Essay prompt text here",
            "word_limit": 500,
            "rubric": {{
                "content": "Evaluation criteria for content",
                "organization": "Evaluation criteria for organization", 
                "grammar": "Evaluation criteria for grammar",
                "creativity": "Evaluation criteria for creativity"
            }}
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            essay_data = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return essay_data
            
        except Exception as e:
            logger.error(f"Error generating essay prompt: {e}")
            raise Exception(f"Failed to generate essay prompt: {str(e)}")

class CurriculumAssistant:
    """AI Curriculum Assistant for lesson planning and resource curation"""
    
    @staticmethod
    def generate_lesson_plan(subject: str, topic: str, grade_level: str, duration: str) -> Dict:
        """Generate a comprehensive lesson plan"""
        prompt = f"""
        Create a comprehensive, detailed lesson plan for {subject} - {topic} for {grade_level} students.
        Duration: {duration}
        
        IMPORTANT: Provide detailed, comprehensive content for each section. Avoid one-liners and brief descriptions. 
        Each section should contain substantial, actionable information that teachers can immediately implement.
        
        Return the response as JSON with the following structure:
        {{
            "lesson_title": "A compelling, descriptive lesson title that captures the essence of the learning experience",
            "learning_objectives": [
                "Students will be able to [specific action verb] [specific content/skill] by [measurable criteria] with [level of accuracy/quality]",
                "Students will demonstrate [specific competency] through [specific assessment method] showing [expected level of mastery]",
                "Students will apply [specific concept/skill] to [real-world context] by [specific application method]"
            ],
            "curriculum_standards": [
                {{
                    "standard_code": "CCSS.MATH.CONTENT.8.EE.A.1",
                    "description": "Comprehensive description of the standard including what students should know and be able to do, with specific learning outcomes and performance indicators",
                    "subject": "Mathematics",
                    "alignment_notes": "Detailed explanation of how this lesson specifically addresses and meets the requirements of this standard"
                }}
            ],
            "materials_needed": [
                "Specific materials with quantities and specifications (e.g., '25 sheets of graph paper, 8.5x11 inch, 4 squares per inch')",
                "Technology requirements with specific software versions or hardware specifications",
                "Handouts and resources with detailed descriptions of content and format"
            ],
            "lesson_structure": [
                {{
                    "phase": "Introduction",
                    "duration": "10 minutes",
                    "activities": [
                        "Detailed step-by-step activity description including teacher actions, student responses, and expected outcomes. Include specific questions to ask, materials to use, and how to transition between activities",
                        "Comprehensive activity description with clear instructions, timing, and assessment criteria for student engagement and understanding"
                    ],
                    "teacher_notes": "Extensive notes including classroom management strategies, potential challenges and solutions, differentiation approaches, and specific pedagogical techniques to employ during this phase"
                }},
                {{
                    "phase": "Main Activity",
                    "duration": "25 minutes",
                    "activities": [
                        "Comprehensive main activity description including detailed procedures, student grouping strategies, specific instructions for each step, expected student behaviors, and assessment checkpoints throughout the activity"
                    ],
                    "teacher_notes": "Detailed implementation guidance including classroom setup, student grouping strategies, potential misconceptions to address, scaffolding techniques, and formative assessment opportunities"
                }},
                {{
                    "phase": "Conclusion",
                    "duration": "10 minutes",
                    "activities": [
                        "Detailed wrap-up activity description including specific reflection questions, summary techniques, student sharing protocols, and connection to future learning"
                    ],
                    "teacher_notes": "Comprehensive assessment and reflection guidance including specific questions to ask students, methods for checking understanding, homework assignments, and preparation notes for the next lesson"
                }}
            ],
            "assessment_strategies": [
                "Detailed formative assessment method including specific questions, observation criteria, student response expectations, and how to use the data to adjust instruction",
                "Comprehensive summative assessment method with specific criteria, rubrics, scoring guidelines, and examples of expected student work"
            ],
            "differentiation_strategies": [
                "Detailed strategy for advanced learners including specific modifications, extension activities, additional resources, and assessment adjustments with clear implementation steps",
                "Comprehensive strategy for struggling learners including specific scaffolding techniques, modified materials, additional support structures, and assessment accommodations with step-by-step guidance"
            ],
            "cross_curricular_connections": [
                {{
                    "subject": "Science",
                    "connection": "Detailed explanation of how this lesson connects to science concepts, including specific scientific principles, real-world applications, and integrated learning opportunities with concrete examples"
                }}
            ],
            "estimated_prep_time": "Detailed breakdown of preparation time including specific tasks, material gathering, technology setup, and classroom preparation requirements"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            lesson_plan = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return lesson_plan
            
        except Exception as e:
            logger.error(f"Error generating lesson plan: {e}")
            raise Exception(f"Failed to generate lesson plan: {str(e)}")
    
    @staticmethod
    def map_curriculum_standards(subject: str, topic: str, grade_level: str) -> Dict:
        """Map curriculum standards for a specific topic"""
        prompt = f"""
        Create a comprehensive mapping of curriculum standards for {subject} - {topic} for {grade_level} students.
        
        IMPORTANT: Provide detailed, comprehensive content for each section. Avoid one-liners and brief descriptions. 
        Each section should contain substantial, actionable information that educators can use for planning and assessment.
        
        Return the response as JSON with the following structure:
        {{
            "subject": "{subject}",
            "topic": "{topic}",
            "grade_level": "{grade_level}",
            "standards": [
                {{
                    "standard_code": "CCSS.MATH.CONTENT.8.EE.A.1",
                    "standard_title": "Comprehensive standard title that clearly identifies the specific learning area and competency level",
                    "description": "Detailed, comprehensive description of the standard including specific learning outcomes, performance expectations, and what students should know and be able to do. Include examples of what mastery looks like and common misconceptions to address",
                    "proficiency_levels": [
                        "Basic understanding: Detailed description of foundational knowledge and skills students must demonstrate, including specific indicators and examples of basic competency",
                        "Proficient application: Comprehensive explanation of how students apply knowledge in various contexts, including specific assessment criteria and real-world application examples",
                        "Advanced mastery: Detailed description of sophisticated understanding and application, including extension activities, higher-order thinking skills, and advanced problem-solving scenarios"
                    ],
                    "assessment_criteria": [
                        "Detailed criterion for assessment including specific observable behaviors, measurable outcomes, and clear indicators of student achievement with examples of what meets, exceeds, and falls below expectations",
                        "Comprehensive assessment criterion with multiple dimensions of evaluation, including content knowledge, skill application, critical thinking, and communication abilities with specific rubrics and scoring guidelines"
                    ],
                    "instructional_implications": "Detailed guidance on how to teach this standard effectively, including recommended instructional strategies, common challenges, differentiation approaches, and formative assessment opportunities"
                }}
            ],
            "learning_progression": [
                {{
                    "prerequisite_skills": ["Detailed description of specific foundational skills students need, including why they are essential and how to assess if students have mastered them", "Comprehensive list of prerequisite knowledge with specific examples and assessment methods"],
                    "current_skills": ["Detailed description of skills students are developing in this topic, including specific learning objectives and performance indicators", "Comprehensive explanation of current learning goals with specific examples and assessment criteria"],
                    "next_skills": ["Detailed description of skills students will develop next, including how current learning prepares them for future topics", "Comprehensive explanation of future learning objectives with specific connections to current instruction"]
                }}
            ],
            "cross_references": [
                {{
                    "related_subject": "Science",
                    "related_topic": "Specific related topic with detailed explanation of the connection",
                    "connection": "Comprehensive explanation of how these subjects and topics connect, including specific concepts, skills, and real-world applications that bridge the disciplines. Include specific examples of integrated learning opportunities and cross-curricular projects"
                }}
            ],
            "implementation_guidance": "Detailed guidance on implementing these standards effectively, including recommended instructional approaches, assessment strategies, differentiation techniques, and common pitfalls to avoid"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.6
            )
            
            content = response.choices[0].message.content
            standards_map = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return standards_map
            
        except Exception as e:
            logger.error(f"Error mapping curriculum standards: {e}")
            raise Exception(f"Failed to map curriculum standards: {str(e)}")
    
    @staticmethod
    def curate_educational_resources(subject: str, topic: str, grade_level: str, resource_types: List[str] = None) -> Dict:
        """Curate educational resources for a topic"""
        if resource_types is None:
            resource_types = ["videos", "articles", "interactive", "worksheets"]
        
        prompt = f"""
        Create a comprehensive curation of educational resources for {subject} - {topic} for {grade_level} students.
        Resource types needed: {', '.join(resource_types)}
        
        IMPORTANT: Provide detailed, comprehensive content for each section. Avoid one-liners and brief descriptions. 
        Each section should contain substantial, actionable information that educators can use to select and implement resources effectively.
        
        Return the response as JSON with the following structure:
        {{
            "subject": "{subject}",
            "topic": "{topic}",
            "grade_level": "{grade_level}",
            "curated_resources": [
                {{
                    "title": "Comprehensive, descriptive resource title that clearly indicates the content and learning focus",
                    "type": "video",
                    "url": "https://example.com/resource",
                    "description": "Detailed, comprehensive description of the resource including specific content covered, learning approach, key concepts addressed, and how it supports the curriculum objectives. Include specific examples of what students will learn and how the resource engages different learning styles",
                    "grade_appropriateness": "Detailed explanation of why this resource is appropriate for the specified grade level, including cognitive development considerations, prerequisite knowledge requirements, and alignment with grade-level standards",
                    "learning_objectives": ["Specific, measurable learning objective that this resource directly addresses, including expected student outcomes and assessment criteria", "Comprehensive learning objective with detailed explanation of how the resource supports achievement of this goal"],
                    "engagement_level": "Detailed assessment of engagement level including specific factors that contribute to student interest, potential challenges to engagement, and strategies to maximize student participation and learning",
                    "accessibility_features": ["Comprehensive list of accessibility features with detailed explanations of how each feature supports different learning needs", "Detailed accessibility considerations including modifications for students with specific learning challenges and universal design principles"],
                    "estimated_duration": "Detailed breakdown of time requirements including setup time, actual resource duration, follow-up activities, and assessment time",
                    "implementation_guidance": "Comprehensive guidance on how to effectively implement this resource in the classroom, including pre-activity preparation, during-activity facilitation, and post-activity assessment and reflection",
                    "differentiation_suggestions": "Detailed suggestions for adapting this resource for different learning needs, including modifications for advanced learners, struggling students, and students with specific learning challenges"
                }}
            ],
            "resource_categories": {{
                "primary_resources": ["Comprehensive list of primary resource URLs with detailed descriptions of why each is essential for core learning objectives and how they should be prioritized in instruction"],
                "supplementary_materials": ["Detailed list of supplementary URLs with specific explanations of how each supports and extends the primary learning objectives, including when and how to use them"],
                "assessment_tools": ["Comprehensive list of assessment URLs with detailed descriptions of assessment types, implementation strategies, and how to use results to inform instruction"],
                "extension_activities": ["Detailed list of extension URLs with comprehensive descriptions of how each activity extends learning, promotes higher-order thinking, and provides opportunities for student choice and creativity"]
            }},
            "curation_notes": "Comprehensive notes about the curation process including specific criteria used for selection, quality indicators, alignment with standards, and recommendations for implementation sequence and timing",
            "last_updated": "2024-01-15",
            "quality_assessment": "Detailed assessment of resource quality including accuracy, relevance, engagement potential, and alignment with best practices in educational technology and pedagogy",
            "implementation_timeline": "Comprehensive timeline for implementing these resources effectively, including recommended sequence, time allocations, and integration with other instructional activities"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            resources = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return resources
            
        except Exception as e:
            logger.error(f"Error curating resources: {e}")
            raise Exception(f"Failed to curate resources: {str(e)}")
    
    @staticmethod
    def create_cross_curricular_unit(subjects: List[str], central_theme: str, grade_level: str) -> Dict:
        """Create a cross-curricular unit integrating multiple subjects"""
        prompt = f"""
        Create a comprehensive cross-curricular unit plan integrating {', '.join(subjects)} around the theme: {central_theme} for {grade_level} students.
        
        IMPORTANT: Provide detailed, comprehensive content for each section. Avoid one-liners and brief descriptions. 
        Each section should contain substantial, actionable information that educators can use to implement a rich, integrated learning experience.
        
        Return the response as JSON with the following structure:
        {{
            "unit_title": "A compelling, descriptive unit title that captures the interdisciplinary nature and central theme of the learning experience",
            "central_theme": "{central_theme}",
            "subjects": {subjects},
            "grade_level": "{grade_level}",
            "unit_overview": "Comprehensive, detailed overview of the unit including the central theme, how subjects integrate, learning goals, and the overall educational approach. Include specific examples of interdisciplinary connections and real-world applications",
            "essential_questions": [
                "Comprehensive essential question that drives inquiry across all subjects, including specific sub-questions and how they connect to different disciplines",
                "Detailed essential question that promotes critical thinking and synthesis of knowledge from multiple subject areas with specific examples of how students will explore this question"
            ],
            "subject_integrations": [
                {{
                    "subject": "Mathematics",
                    "lesson_focus": "Detailed explanation of how mathematics connects to the central theme, including specific mathematical concepts, skills, and applications that support the interdisciplinary learning goals",
                    "key_concepts": ["Comprehensive description of key mathematical concepts including definitions, importance, and real-world applications", "Detailed explanation of mathematical principles and their relevance to the central theme"],
                    "activities": ["Detailed, step-by-step activity description including specific mathematical procedures, student engagement strategies, and assessment checkpoints", "Comprehensive activity description with clear learning objectives, materials needed, and expected student outcomes"],
                    "assessment": "Detailed assessment method including specific criteria, rubrics, and examples of expected student work, with clear indicators of mathematical understanding and application"
                }}
            ],
            "project_based_learning": {{
                "project_title": "Comprehensive, descriptive project title that clearly indicates the interdisciplinary nature and learning outcomes",
                "project_description": "Detailed, comprehensive project description including specific objectives, procedures, student roles, timeline, and expected outcomes. Include specific examples of how the project integrates all subject areas",
                "learning_outcomes": ["Specific, measurable learning outcome with detailed criteria for success and assessment methods", "Comprehensive learning outcome including both content knowledge and skill development with specific indicators of achievement"],
                "timeline": "Detailed timeline with specific milestones, checkpoints, and deadlines including daily or weekly breakdown of activities and responsibilities",
                "materials_needed": ["Comprehensive list of materials with specific quantities, specifications, and sources", "Detailed materials list including technology requirements, consumables, and any special equipment needed"],
                "assessment_rubric": {{
                    "criteria": ["Detailed assessment criterion with specific indicators of quality and performance expectations", "Comprehensive assessment criterion including multiple dimensions of evaluation with clear scoring guidelines"],
                    "scoring": "Detailed scoring system with specific point allocations, quality descriptors, and examples of work at different achievement levels"
                }},
                "implementation_guidance": "Comprehensive guidance on implementing the project including classroom management strategies, student grouping approaches, technology integration, and differentiation techniques"
            }},
            "weekly_schedule": [
                {{
                    "week": 1,
                    "focus": "Detailed focus description including specific learning objectives, subject integrations, and key activities for the week",
                    "activities": ["Comprehensive activity description with specific procedures, materials, and assessment methods", "Detailed activity description including student engagement strategies, differentiation approaches, and connection to the central theme"]
                }}
            ],
            "assessment_strategies": [
                "Detailed formative assessment method including specific strategies, tools, and criteria for monitoring student progress throughout the unit",
                "Comprehensive summative assessment method with specific rubrics, scoring guidelines, and examples of expected student work across all subject areas"
            ],
            "estimated_duration": "Detailed breakdown of unit duration including specific time allocations for each subject area, project work, and assessment activities",
            "differentiation_strategies": "Comprehensive differentiation strategies including modifications for different learning styles, ability levels, and special needs with specific examples and implementation guidance",
            "technology_integration": "Detailed technology integration plan including specific tools, software, and digital resources that support interdisciplinary learning and project completion"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.8
            )
            
            content = response.choices[0].message.content
            unit_plan = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return unit_plan
            
        except Exception as e:
            logger.error(f"Error creating cross-curricular unit: {e}")
            raise Exception(f"Failed to create cross-curricular unit: {str(e)}")

class AssessmentGrader:
    """Grade assessments using OpenAI"""
    
    @staticmethod
    def grade_multiple_choice(questions: List[Dict], student_answers: List[str]) -> Dict:
        """Grade multiple choice questions"""
        correct_count = 0
        total_questions = len(questions)
        detailed_results = []
        
        for i, (question, student_answer) in enumerate(zip(questions, student_answers)):
            is_correct = student_answer.upper() == question['correct_answer']
            if is_correct:
                correct_count += 1
            
            detailed_results.append({
                "question_number": i + 1,
                "student_answer": student_answer,
                "correct_answer": question['correct_answer'],
                "is_correct": is_correct,
                "explanation": question['explanation']
            })
        
        score_percentage = (correct_count / total_questions) * 100
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "score_percentage": round(score_percentage, 2),
            "detailed_results": detailed_results,
            "grade": AssessmentGrader._get_letter_grade(score_percentage)
        }
    
    @staticmethod
    def grade_essay(essay_prompt: Dict, student_essay: str) -> Dict:
        """Grade essay using AI"""
        prompt = f"""
        Provide a comprehensive, detailed assessment of the following essay based on the provided rubric.
        
        Essay Prompt: {essay_prompt['prompt']}
        Word Limit: {essay_prompt['word_limit']} words
        
        Rubric:
        - Content: {essay_prompt['rubric']['content']}
        - Organization: {essay_prompt['rubric']['organization']}
        - Grammar: {essay_prompt['rubric']['grammar']}
        - Creativity: {essay_prompt['rubric']['creativity']}
        
        Student Essay:
        {student_essay}
        
        IMPORTANT: Provide detailed, comprehensive feedback for each section. Avoid one-liners and brief descriptions. 
        Each feedback section should contain substantial, actionable information that students can use to improve their writing.
        
        Return the response as JSON with the following structure:
        {{
            "overall_score": 85,
            "word_count": 450,
            "detailed_scores": {{
                "content": {{
                    "score": 20,
                    "max_score": 25,
                    "feedback": "Comprehensive, detailed feedback on content including specific strengths and areas for improvement. Include specific examples from the essay, suggestions for enhancement, and clear guidance on how to address content-related issues. Provide actionable advice for strengthening arguments, supporting evidence, and developing ideas more fully"
                }},
                "organization": {{
                    "score": 18,
                    "max_score": 25,
                    "feedback": "Detailed feedback on organization including analysis of structure, flow, and coherence. Include specific comments on paragraph organization, transitions, logical progression, and overall essay structure. Provide concrete suggestions for improving organization and creating better flow between ideas"
                }},
                "grammar": {{
                    "score": 22,
                    "max_score": 25,
                    "feedback": "Comprehensive feedback on grammar, mechanics, and language use including specific error patterns, corrections, and suggestions for improvement. Include detailed explanations of grammatical concepts, examples of correct usage, and strategies for avoiding common errors"
                }},
                "creativity": {{
                    "score": 20,
                    "max_score": 25,
                    "feedback": "Detailed feedback on creativity including analysis of original thinking, unique perspectives, and innovative approaches. Include specific comments on creative elements, suggestions for enhancing originality, and guidance on developing more engaging and imaginative content"
                }}
            }},
            "overall_feedback": "Comprehensive, detailed overall feedback on the essay including overall strengths, areas for improvement, and specific recommendations for enhancement. Include analysis of how well the essay addresses the prompt, meets the word limit, and demonstrates understanding of the topic. Provide clear guidance on next steps for improvement",
            "grade": "B+",
            "strengths_analysis": "Detailed analysis of the essay's strengths including specific examples of effective writing, strong arguments, and well-developed ideas. Highlight what the student did particularly well and why these elements are effective",
            "improvement_areas": "Comprehensive identification of areas for improvement including specific suggestions for enhancement, examples of how to implement changes, and prioritized recommendations for the most impactful improvements",
            "next_steps": "Detailed guidance on next steps for improvement including specific actions the student can take, resources they might consult, and practice activities that would help develop their writing skills"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            grading_result = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return grading_result
            
        except Exception as e:
            logger.error(f"Error grading essay: {e}")
            raise Exception(f"Failed to grade essay: {str(e)}")
    
    @staticmethod
    def _get_letter_grade(percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 93:
            return "A"
        elif percentage >= 90:
            return "A-"
        elif percentage >= 87:
            return "B+"
        elif percentage >= 83:
            return "B"
        elif percentage >= 80:
            return "B-"
        elif percentage >= 77:
            return "C+"
        elif percentage >= 73:
            return "C"
        elif percentage >= 70:
            return "C-"
        elif percentage >= 67:
            return "D+"
        elif percentage >= 63:
            return "D"
        elif percentage >= 60:
            return "D-"
        else:
            return "F"

class AnnouncementGenerator:
    """Generate enhanced announcements with professional tone and context management"""
    
    def __init__(self):
        # Test database connection on initialization
        if not db_manager.test_connection():
            logger.warning("Database connection failed. Context will be limited.")
    
    def add_to_context(self, announcement_type: str, original_title: str, original_message: str, 
                      enhanced_title: str, enhanced_message: str, target_audience: str, 
                      tone: str, context_connections: str = None):
        """Add announcement to database context for better continuity"""
        try:
            success = db_manager.add_announcement_context(
                announcement_type=announcement_type,
                original_title=original_title,
                original_message=original_message,
                enhanced_title=enhanced_title,
                enhanced_message=enhanced_message,
                target_audience=target_audience,
                tone=tone,
                context_connections=context_connections
            )
            if success:
                logger.info(f"Added announcement to context: {enhanced_title[:50]}...")
            else:
                logger.error("Failed to add announcement to context")
        except Exception as e:
            logger.error(f"Error adding to context: {e}")
    
    def get_context_summary(self) -> str:
        """Get a summary of recent announcements for context from database"""
        try:
            return db_manager.get_context_summary(limit=3)
        except Exception as e:
            logger.error(f"Error getting context summary: {e}")
            return "No previous announcements."
    
    def create_enhanced_announcement(self, title: str, message_body: str, announcement_type: str = "General Notice", 
                                   target_audience: str = "All", tone: str = "Professional", 
                                   max_title_length: int = 20, max_message_length: int = 400) -> Dict:
        """Create an enhanced announcement with improved title and professional message body"""
        
        # Validate announcement type
        valid_announcement_types = [
            "General Notice", "Academic Update", "Events", "Holiday/Closure", "PTM/Meetings"
        ]
        if announcement_type not in valid_announcement_types:
            announcement_type = "General Notice"  # Default fallback
        
        # Validate target audience
        valid_target_audiences = [
            "All", 
            "Staff Members - Primary Wing", "Staff Members - Secondary Wing",
            "Teachers - Primary Wing", "Teachers - Secondary Wing", 
            "Family - Primary Wing", "Family - Secondary Wing"
        ]
        if target_audience not in valid_target_audiences:
            target_audience = "All"  # Default fallback
        
        # Build context-aware prompt
        context_summary = self.get_context_summary()
        
        prompt = f"""
        Create THREE different enhanced announcement options for a SCHOOL environment. This is for school administrators to communicate with staff, teachers, and families. Consider the context of recent announcements to maintain consistency and avoid repetition.

        CONTEXT:
        {context_summary}

        INPUT INFORMATION:
        - Original Title: {title}
        - Original Message: {message_body}
        - Announcement Type: {announcement_type}
        - Target Audience: {target_audience}
        - Desired Tone: {tone}

        REQUIREMENTS:
        1. CREATE 3 DIFFERENT OPTIONS: Each with unique approach and style
        2. TITLE LIMITS: Maximum {max_title_length} characters per title - make them concise, clear, and school-appropriate
        3. MESSAGE BODY LIMITS: Maximum {max_message_length} characters per message - include key information, clear instructions, and school-appropriate tone
        4. MAINTAIN CONSISTENCY: Ensure all options align with recent announcements in tone and style
        5. SCHOOL-APPROPRIATE TONE: Use language suitable for school environment (not corporate/business)
        6. CONTEXT AWARENESS: Reference or build upon previous announcements when relevant

        SCHOOL-SPECIFIC GUIDELINES:
        - Use "Dear Parents", "Dear Teachers", "Dear Staff" instead of "Dear Team"
        - Use "school", "students", "parents", "teachers" instead of corporate terms
        - Keep tone warm, informative, and educational
        - Focus on student welfare, academic progress, and school community
        - Avoid business jargon like "quarterly reviews", "performance metrics", "stakeholders"

        Return the response as JSON with the following structure:
        {{
            "options": [
                {{
                    "title": "Title option 1 (max {max_title_length} chars)",
                    "message": "Message option 1 (max {max_message_length} chars)"
                }},
                {{
                    "title": "Title option 2 (max {max_title_length} chars)", 
                    "message": "Message option 2 (max {max_message_length} chars)"
                }},
                {{
                    "title": "Title option 3 (max {max_title_length} chars)",
                    "message": "Message option 3 (max {max_message_length} chars)"
                }}
            ]
        }}

        IMPORTANT: Always put "title" first, then "message" in each option object.

        IMPORTANT: 
        - Each title must be exactly {max_title_length} characters or less
        - Each message must be exactly {max_message_length} characters or less
        - Provide 3 distinctly different approaches
        - Use school-appropriate language and tone
        - Focus on educational context, not business context
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            # Clean the content to remove any invalid JSON characters
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            try:
                enhanced_announcement = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                # Fallback to basic structure
                enhanced_announcement = {
                    "options": [
                        {
                            "title": f"Enhanced: {title[:17]}",
                            "message": f"Enhanced message: {message_body[:380]}"
                        },
                        {
                            "title": f"Update: {title[:15]}",
                            "message": f"Updated announcement: {message_body[:380]}"
                        },
                        {
                            "title": f"Notice: {title[:15]}",
                            "message": f"Important notice: {message_body[:380]}"
                        }
                    ]
                }
            
            # Ensure title comes before message in each option
            if 'options' in enhanced_announcement:
                reordered_options = []
                for option in enhanced_announcement['options']:
                    if 'title' in option and 'message' in option:
                        # Create new dict with title first
                        reordered_option = {
                            'title': option['title'],
                            'message': option['message']
                        }
                        reordered_options.append(reordered_option)
                    else:
                        reordered_options.append(option)
                enhanced_announcement['options'] = reordered_options
            
            # Add to database context (use first option as primary)
            if 'options' in enhanced_announcement and len(enhanced_announcement['options']) > 0:
                first_option = enhanced_announcement['options'][0]
                self.add_to_context(
                    announcement_type=announcement_type,
                    original_title=title,
                    original_message=message_body,
                    enhanced_title=first_option['title'],
                    enhanced_message=first_option['message'],
                    target_audience=target_audience,
                    tone=tone,
                    context_connections=None
                )
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return enhanced_announcement
            
        except Exception as e:
            logger.error(f"Error creating enhanced announcement: {e}")
            raise Exception(f"Failed to create enhanced announcement: {str(e)}")
    
    def get_announcement_suggestions(self, announcement_type: str, target_audience: str) -> Dict:
        """Get suggestions for announcement structure and content based on type and audience"""
        
        prompt = f"""
        Provide focused suggestions for creating effective announcements for:
        - Type: {announcement_type}
        - Target Audience: {target_audience}

        Consider the context of recent announcements:
        {self.get_context_summary()}

        IMPORTANT: Provide exactly 3 suggestions for each category. Keep suggestions concise and actionable.

        Return the response as JSON with the following structure:
        {{
            "title_suggestions": [
                "First title suggestion for {announcement_type} announcements",
                "Second title suggestion with alternative approach",
                "Third title suggestion following best practices"
            ],
            "message_body_suggestions": [
                "First message structure suggestion for {target_audience}",
                "Second message content approach",
                "Third message formatting recommendation"
            ],
            "tone_suggestions": [
                "First tone recommendation for {target_audience}",
                "Second language style suggestion",
                "Third communication approach"
            ],
            "context_considerations": "How to build upon or reference previous announcements effectively",
            "audience_specific_notes": "Tailored recommendations for the specific target audience"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.6
            )
            
            content = response.choices[0].message.content
            suggestions = json.loads(content)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting announcement suggestions: {e}")
            raise Exception(f"Failed to get announcement suggestions: {str(e)}")
    
    def clear_context(self):
        """Clear the context history from database"""
        try:
            success = db_manager.clear_context()
            if success:
                return {"message": "Context history cleared successfully from database"}
            else:
                return {"message": "Failed to clear context history"}
        except Exception as e:
            logger.error(f"Error clearing context: {e}")
            return {"message": f"Error clearing context: {str(e)}"}

# Import database manager
from database_manager import db_manager

# Initialize announcement generator
announcement_generator = AnnouncementGenerator()

# API Routes

@app.route('/')
def index():
    """Serve the main UI interface"""
    return render_template('index.html')

@app.route('/static/swagger.json')
def swagger_json():
    """Serve Swagger JSON file"""
    return app.send_static_file('swagger.json')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Automated Assessment API"
    })

@app.route('/api/token-usage', methods=['GET'])
def get_token_usage():
    """Get current token usage statistics"""
    return jsonify(token_tracker.get_daily_usage())

@app.route('/api/generate/multiple-choice', methods=['POST'])
def generate_multiple_choice():
    """Generate multiple choice questions"""
    try:
        data = request.get_json()
        subject = data.get('subject', 'General')
        topic = data.get('topic', 'General Knowledge')
        num_questions = data.get('num_questions', 5)
        
        # Check token limit
        estimated_tokens = num_questions * 200  # Rough estimate
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        questions = AssessmentGenerator.generate_multiple_choice_questions(
            subject, topic, num_questions
        )
        
        return jsonify({
            "success": True,
            "questions": questions,
            "metadata": {
                "subject": subject,
                "topic": topic,
                "num_questions": num_questions,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in generate_multiple_choice: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate/essay-prompt', methods=['POST'])
def generate_essay_prompt():
    """Generate essay prompt"""
    try:
        data = request.get_json()
        subject = data.get('subject', 'General')
        topic = data.get('topic', 'General Knowledge')
        
        # Check token limit
        estimated_tokens = 300  # Rough estimate
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        essay_data = AssessmentGenerator.generate_essay_prompt(subject, topic)
        
        return jsonify({
            "success": True,
            "essay_prompt": essay_data,
            "metadata": {
                "subject": subject,
                "topic": topic,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in generate_essay_prompt: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/grade/multiple-choice', methods=['POST'])
def grade_multiple_choice():
    """Grade multiple choice questions"""
    try:
        data = request.get_json()
        questions = data.get('questions', [])
        student_answers = data.get('student_answers', [])
        
        if len(questions) != len(student_answers):
            return jsonify({"error": "Number of questions and answers must match"}), 400
        
        grading_result = AssessmentGrader.grade_multiple_choice(questions, student_answers)
        
        return jsonify({
            "success": True,
            "grading_result": grading_result,
            "metadata": {
                "graded_at": datetime.now().isoformat(),
                "total_questions": len(questions)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in grade_multiple_choice: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/grade/essay', methods=['POST'])
def grade_essay():
    """Grade essay"""
    try:
        data = request.get_json()
        essay_prompt = data.get('essay_prompt', {})
        student_essay = data.get('student_essay', '')
        
        if not student_essay.strip():
            return jsonify({"error": "Student essay cannot be empty"}), 400
        
        # Check token limit
        estimated_tokens = len(student_essay.split()) * 2  # Rough estimate
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        grading_result = AssessmentGrader.grade_essay(essay_prompt, student_essay)
        
        return jsonify({
            "success": True,
            "grading_result": grading_result,
            "metadata": {
                "graded_at": datetime.now().isoformat(),
                "word_count": len(student_essay.split())
            }
        })
        
    except Exception as e:
        logger.error(f"Error in grade_essay: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/assessment/complete', methods=['POST'])
def create_complete_assessment():
    """Create a complete assessment with questions and grading"""
    try:
        data = request.get_json()
        subject = data.get('subject', 'General')
        topic = data.get('topic', 'General Knowledge')
        assessment_type = data.get('type', 'multiple_choice')  # 'multiple_choice' or 'essay'
        
        if assessment_type == 'multiple_choice':
            # Generate questions
            questions = AssessmentGenerator.generate_multiple_choice_questions(subject, topic)
            
            return jsonify({
                "success": True,
                "assessment": {
                    "type": "multiple_choice",
                    "questions": questions,
                    "metadata": {
                        "subject": subject,
                        "topic": topic,
                        "created_at": datetime.now().isoformat()
                    }
                }
            })
            
        elif assessment_type == 'essay':
            # Generate essay prompt
            essay_data = AssessmentGenerator.generate_essay_prompt(subject, topic)
            
            return jsonify({
                "success": True,
                "assessment": {
                    "type": "essay",
                    "essay_prompt": essay_data,
                    "metadata": {
                        "subject": subject,
                        "topic": topic,
                        "created_at": datetime.now().isoformat()
                    }
                }
            })
        
        else:
            return jsonify({"error": "Invalid assessment type. Use 'multiple_choice' or 'essay'"}), 400
            
    except Exception as e:
        logger.error(f"Error in create_complete_assessment: {e}")
        return jsonify({"error": str(e)}), 500

# Curriculum Assistant API Routes

@app.route('/api/curriculum/generate-lesson-plan', methods=['POST'])
def generate_lesson_plan():
    """Generate a comprehensive lesson plan"""
    try:
        data = request.get_json()
        subject = data.get('subject')
        topic = data.get('topic')
        grade_level = data.get('grade_level')
        duration = data.get('duration')
        
        # Validate required parameters
        if not subject or not topic or not grade_level or not duration:
            return jsonify({
                "error": "Missing required parameters. Please provide: subject, topic, grade_level, and duration"
            }), 400
        
        # Validate duration format
        duration_lower = duration.lower()
        valid_units = ['minutes', 'hours', 'days', 'weeks', 'months']
        if not any(unit in duration_lower for unit in valid_units):
            return jsonify({
                "error": "Invalid duration format. Please use format like '45 minutes', '2 hours', '3 days', '1 week', or '2 months'"
            }), 400
        
        # Check token limit
        estimated_tokens = 800  # Rough estimate for lesson plan generation
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        lesson_plan = CurriculumAssistant.generate_lesson_plan(
            subject, topic, grade_level, duration
        )
        
        return jsonify({
            "success": True,
            "lesson_plan": lesson_plan,
            "metadata": {
                "subject": subject,
                "topic": topic,
                "grade_level": grade_level,
                "duration": duration,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in generate_lesson_plan: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/curriculum/map-standards', methods=['POST'])
def map_curriculum_standards():
    """Map curriculum standards for a specific topic"""
    try:
        data = request.get_json()
        subject = data.get('subject', 'General')
        topic = data.get('topic', 'General Knowledge')
        grade_level = data.get('grade_level', '8th grade')
        
        # Check token limit
        estimated_tokens = 600  # Rough estimate for standards mapping
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        standards_map = CurriculumAssistant.map_curriculum_standards(
            subject, topic, grade_level
        )
        
        return jsonify({
            "success": True,
            "standards_map": standards_map,
            "metadata": {
                "subject": subject,
                "topic": topic,
                "grade_level": grade_level,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in map_curriculum_standards: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/curriculum/curate-resources', methods=['POST'])
def curate_educational_resources():
    """Curate educational resources for a topic"""
    try:
        data = request.get_json()
        subject = data.get('subject', 'General')
        topic = data.get('topic', 'General Knowledge')
        grade_level = data.get('grade_level', '8th grade')
        resource_types = data.get('resource_types', ['videos', 'articles', 'interactive', 'worksheets'])
        
        # Check token limit
        estimated_tokens = 700  # Rough estimate for resource curation
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        resources = CurriculumAssistant.curate_educational_resources(
            subject, topic, grade_level, resource_types
        )
        
        return jsonify({
            "success": True,
            "resources": resources,
            "metadata": {
                "subject": subject,
                "topic": topic,
                "grade_level": grade_level,
                "resource_types": resource_types,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in curate_educational_resources: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/curriculum/cross-curricular-unit', methods=['POST'])
def create_cross_curricular_unit():
    """Create a cross-curricular unit integrating multiple subjects"""
    try:
        data = request.get_json()
        subjects = data.get('subjects', ['Mathematics', 'Science'])
        central_theme = data.get('central_theme', 'Environmental Sustainability')
        grade_level = data.get('grade_level', '8th grade')
        
        # Check token limit
        estimated_tokens = 1000  # Rough estimate for cross-curricular unit
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        unit_plan = CurriculumAssistant.create_cross_curricular_unit(
            subjects, central_theme, grade_level
        )
        
        return jsonify({
            "success": True,
            "unit_plan": unit_plan,
            "metadata": {
                "subjects": subjects,
                "central_theme": central_theme,
                "grade_level": grade_level,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in create_cross_curricular_unit: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/curriculum/complete-package', methods=['POST'])
def create_complete_curriculum_package():
    """Create a complete curriculum package with lesson plan, standards, and resources"""
    try:
        data = request.get_json()
        subject = data.get('subject')
        topic = data.get('topic')
        grade_level = data.get('grade_level')
        duration = data.get('duration')
        include_resources = data.get('include_resources', True)
        
        # Validate required parameters
        if not subject or not topic or not grade_level or not duration:
            return jsonify({
                "error": "Missing required parameters. Please provide: subject, topic, grade_level, and duration"
            }), 400
        
        # Validate duration format
        duration_lower = duration.lower()
        valid_units = ['minutes', 'hours', 'days', 'weeks', 'months']
        if not any(unit in duration_lower for unit in valid_units):
            return jsonify({
                "error": "Invalid duration format. Please use format like '45 minutes', '2 hours', '3 days', '1 week', or '2 months'"
            }), 400
        
        # Check token limit for complete package
        estimated_tokens = 1500  # Rough estimate for complete package
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        # Generate lesson plan
        lesson_plan = CurriculumAssistant.generate_lesson_plan(
            subject, topic, grade_level, duration
        )
        
        # Map standards
        standards_map = CurriculumAssistant.map_curriculum_standards(
            subject, topic, grade_level
        )
        
        # Curate resources if requested
        resources = None
        if include_resources:
            resources = CurriculumAssistant.curate_educational_resources(
                subject, topic, grade_level
            )
        
        return jsonify({
            "success": True,
            "curriculum_package": {
                "lesson_plan": lesson_plan,
                "standards_map": standards_map,
                "resources": resources,
                "metadata": {
                    "subject": subject,
                    "topic": topic,
                    "grade_level": grade_level,
                    "duration": duration,
                    "created_at": datetime.now().isoformat()
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in create_complete_curriculum_package: {e}")
        return jsonify({"error": str(e)}), 500

# Announcement API Routes



@app.route('/api/announcement/suggestions', methods=['POST'])
def get_announcement_suggestions():
    """Get suggestions for announcement structure and content"""
    try:
        data = request.get_json()
        announcement_type = data.get('announcement_type', 'General')
        target_audience = data.get('target_audience', 'All Staff')
        
        # Check token limit
        estimated_tokens = 600  # Rough estimate for suggestions
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        suggestions = announcement_generator.get_announcement_suggestions(
            announcement_type, target_audience
        )
        
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "metadata": {
                "announcement_type": announcement_type,
                "target_audience": target_audience,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_announcement_suggestions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/context', methods=['GET'])
def get_announcement_context():
    """Get current announcement context history from database"""
    try:
        context_summary = announcement_generator.get_context_summary()
        recent_announcements = db_manager.get_recent_announcements(limit=5)
        stats = db_manager.get_announcement_stats()
        
        return jsonify({
            "success": True,
            "context_summary": context_summary,
            "context_count": stats["total_announcements"],
            "recent_announcements": recent_announcements,
            "stats": stats,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_announcement_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/context/clear', methods=['POST'])
def clear_announcement_context():
    """Clear the announcement context history from database"""
    try:
        result = announcement_generator.clear_context()
        
        return jsonify({
            "success": True,
            "result": result,
            "metadata": {
                "cleared_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in clear_announcement_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/context/by-type/<announcement_type>', methods=['GET'])
def get_announcements_by_type(announcement_type: str):
    """Get announcements by type from database"""
    try:
        limit = request.args.get('limit', 5, type=int)
        announcements = db_manager.get_announcements_by_type(announcement_type, limit)
        
        return jsonify({
            "success": True,
            "announcement_type": announcement_type,
            "announcements": announcements,
            "count": len(announcements),
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_announcements_by_type: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/context/by-audience/<target_audience>', methods=['GET'])
def get_announcements_by_audience(target_audience: str):
    """Get announcements by target audience from database"""
    try:
        limit = request.args.get('limit', 5, type=int)
        announcements = db_manager.get_announcements_by_audience(target_audience, limit)
        
        return jsonify({
            "success": True,
            "target_audience": target_audience,
            "announcements": announcements,
            "count": len(announcements),
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_announcements_by_audience: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/context/stats', methods=['GET'])
def get_announcement_stats():
    """Get announcement statistics from database"""
    try:
        stats = db_manager.get_announcement_stats()
        
        return jsonify({
            "success": True,
            "stats": stats,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_announcement_stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/context/delete/<int:announcement_id>', methods=['DELETE'])
def delete_announcement(announcement_id: int):
    """Delete specific announcement from database"""
    try:
        success = db_manager.delete_announcement(announcement_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Announcement {announcement_id} deleted successfully",
                "metadata": {
                    "deleted_at": datetime.now().isoformat(),
                    "database_connected": db_manager.test_connection()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to delete announcement {announcement_id}",
                "metadata": {
                    "attempted_at": datetime.now().isoformat(),
                    "database_connected": db_manager.test_connection()
                }
            }), 404
        
    except Exception as e:
        logger.error(f"Error in delete_announcement: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/announcement/create', methods=['POST'])
def create_enhanced_announcement():
    """Create an enhanced announcement with improved title and professional message body"""
    try:
        data = request.get_json()
        title = data.get('title')
        message_body = data.get('message_body')
        announcement_type = data.get('announcement_type', 'General Notice')
        target_audience = data.get('target_audience', 'All')
        tone = data.get('tone', 'Professional')
        max_title_length = data.get('max_title_length', 20)
        max_message_length = data.get('max_message_length', 400)
        
        # Validate required parameters
        if not title or not message_body:
            return jsonify({
                "error": "Missing required parameters. Please provide: title and message_body"
            }), 400
        
        # Check token limit for complete package
        estimated_tokens = 1200  # Rough estimate for complete package
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        # Generate enhanced announcement
        enhanced_announcement = announcement_generator.create_enhanced_announcement(
            title, message_body, announcement_type, target_audience, tone, 
            max_title_length, max_message_length
        )
        
        # Get suggestions if requested
        suggestions = None
        include_suggestions = data.get('include_suggestions', False)
        if include_suggestions:
            suggestions = announcement_generator.get_announcement_suggestions(
                announcement_type, target_audience
            )
        
        # Convert to numbered object format for easy frontend parsing
        response_options = []
        for i, option in enumerate(enhanced_announcement["options"], 1):
            if 'title' in option and 'message' in option:
                # Format: {"title1": "...", "message1": "..."}
                option_object = {
                    f"title{i}": option['title'],
                    f"message{i}": option['message']
                }
                response_options.append(option_object)
            else:
                response_options.append({f"option{i}": str(option)})
        
        # Create response as array of numbered objects
        response_data = {
            "success": True,
            "options": response_options
        }
        
        # Convert to JSON string
        response_json = json.dumps(response_data, separators=(',', ':'))
        return app.response_class(response_json, mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Error in create_complete_announcement_package: {e}")
        return jsonify({"error": str(e)}), 500

# Thread-safe model loading
model_lock = threading.Lock()
model = None
preprocessor = None

def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = load_model('results/best_model_random_forest_pipeline.pkl')
    return model

# If you have a preprocessor.pkl, load it here similarly
# def get_preprocessor():
#     global preprocessor
#     if preprocessor is None:
#         with model_lock:
#             if preprocessor is None:
#                 preprocessor = load_model('results/preprocessor.pkl')
#     return preprocessor

PREDICT_COLUMNS = [
    'User ID', 'Timestamp', 'Login Status', 'IP Address', 'Device Type',
    'Location', 'Session Duration', 'Failed Attempts', 'Behavioral Score'
]

@app.route('/api/predict', methods=['POST'])
def predict_anomaly():
    """Predict anomaly status for input data (single or batch)."""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No input data provided'}), 400

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            return jsonify({'error': 'Input must be a dict or list of dicts'}), 400

        # Validate columns (optional, but you can keep it)
        for record in data:
            missing = [col for col in PREDICT_COLUMNS if col not in record]
            if missing:
                return jsonify({'error': f'Missing columns: {missing}'}), 400

        df = pd.DataFrame(data)
        model = get_model()
        preds = model.predict(df)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[:, 1].tolist()
        else:
            proba = [None] * len(preds)

        response = []
        for i, record in enumerate(data):
            response.append({
                'input': record,
                'predicted_anomaly': int(preds[i]),
                'anomaly_probability': proba[i]
            })
        return jsonify({'predictions': response, 'count': len(response)})
    except Exception as e:
        logger.error(f"Error in predict_anomaly: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
