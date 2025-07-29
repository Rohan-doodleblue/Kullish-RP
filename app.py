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
            
            # Track LLM usage for multiple choice questions generation
            tokens_used = response.usage.total_tokens
            estimated_cost = (tokens_used / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
            track_llm_text_usage(tokens_used, estimated_cost, "multiple_choice_generation")
            
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
            
            # Track LLM usage for essay prompt generation
            tokens_used = response.usage.total_tokens
            estimated_cost = (tokens_used / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
            track_llm_text_usage(tokens_used, estimated_cost, "essay_prompt_generation")
            
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
            
            # Track LLM usage for lesson plan generation
            tokens_used = response.usage.total_tokens
            estimated_cost = (tokens_used / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
            track_llm_text_usage(tokens_used, estimated_cost, "lesson_plan_generation")
            
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
            
            # Track LLM usage for curriculum standards mapping
            tokens_used = response.usage.total_tokens
            estimated_cost = (tokens_used / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
            track_llm_text_usage(tokens_used, estimated_cost, "curriculum_standards_mapping")
            
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
            
            # Track LLM usage for announcement creation
            tokens_used = response.usage.total_tokens
            estimated_cost = (tokens_used / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
            track_llm_text_usage(tokens_used, estimated_cost, "announcement_creation")
            
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
            
            # Track LLM usage for announcement suggestions
            tokens_used = response.usage.total_tokens
            estimated_cost = (tokens_used / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
            track_llm_text_usage(tokens_used, estimated_cost, "announcement_suggestions")
            
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

class TitleEnhancer:
    """Enhanced title generation with context management and unique suggestions"""
    
    def __init__(self):
        # Test database connection on initialization
        if not db_manager.test_connection():
            logger.warning("Database connection failed. Context will be limited.")
    
    def add_title_to_context(self, original_title: str, enhanced_titles: List[str], 
                           context_type: str = "title_enhancement", 
                           domain: str = "general"):
        """Add enhanced titles to database context for better continuity"""
        try:
            success = db_manager.add_title_context(
                original_title=original_title,
                enhanced_titles=enhanced_titles,
                context_type=context_type,
                domain=domain
            )
            if success:
                logger.info(f"Added title enhancement to context: {original_title[:30]}...")
            else:
                logger.error("Failed to add title enhancement to context")
        except Exception as e:
            logger.error(f"Error adding title to context: {e}")
    
    def get_title_context_summary(self, domain: str = "general") -> str:
        """Get a summary of recent title enhancements for context from database"""
        try:
            return db_manager.get_title_context_summary(domain=domain, limit=3)
        except Exception as e:
            logger.error(f"Error getting title context summary: {e}")
            return "No previous title enhancements."
    
    def enhance_title(self, original_title: str, context_type: str = "general", 
                     domain: str = "general", max_length: int = 50, 
                     style_preference: str = "professional") -> Dict:
        """Create enhanced title suggestions with context awareness"""
        
        # Validate context type
        valid_context_types = [
            "general", "academic", "business", "creative", "technical", 
            "marketing", "educational", "professional", "casual"
        ]
        if context_type not in valid_context_types:
            context_type = "general"
        
        # Validate style preference
        valid_styles = [
            "professional", "creative", "concise", "descriptive", 
            "attention-grabbing", "formal", "casual", "technical"
        ]
        if style_preference not in valid_styles:
            style_preference = "professional"
        
        # Build context-aware prompt
        context_summary = self.get_title_context_summary(domain)
        
        prompt = f"""
        Create THREE UNIQUE and DISTINCTLY DIFFERENT enhanced title suggestions for the given original title.
        
        CONTEXT:
        {context_summary}
        
        INPUT INFORMATION:
        - Original Title: "{original_title}"
        - Context Type: {context_type}
        - Domain: {domain}
        - Style Preference: {style_preference}
        - Maximum Length: {max_length} characters
        
        REQUIREMENTS:
        1. CREATE 3 COMPLETELY DIFFERENT APPROACHES: Each suggestion should have a unique style, tone, and approach
        2. CHARACTER LIMIT: Maximum {max_length} characters per title
        3. CONTEXT AWARENESS: Consider recent title enhancements to avoid repetition and maintain consistency
        4. STYLE VARIETY: Provide different stylistic approaches while maintaining the core message
        5. DOMAIN APPROPRIATENESS: Ensure titles are suitable for the specified domain
        
        STYLE GUIDELINES:
        - Professional: Clear, concise, business-appropriate
        - Creative: Engaging, imaginative, attention-grabbing
        - Concise: Short, direct, to-the-point
        - Descriptive: Detailed, informative, comprehensive
        - Attention-grabbing: Bold, compelling, memorable
        - Formal: Structured, academic, official
        - Casual: Friendly, approachable, conversational
        - Technical: Precise, specialized, industry-specific
        
        Return the response as JSON with the following structure:
        {{
            "original_title": "{original_title}",
            "enhanced_titles": [
                {{
                    "title": "First enhanced title suggestion (max {max_length} chars)",
                    "style": "professional",
                    "approach": "Brief description of the approach used",
                    "character_count": 25
                }},
                {{
                    "title": "Second enhanced title suggestion (max {max_length} chars)",
                    "style": "creative", 
                    "approach": "Brief description of the approach used",
                    "character_count": 30
                }},
                {{
                    "title": "Third enhanced title suggestion (max {max_length} chars)",
                    "style": "concise",
                    "approach": "Brief description of the approach used", 
                    "character_count": 20
                }}
            ],
            "metadata": {{
                "context_type": "{context_type}",
                "domain": "{domain}",
                "style_preference": "{style_preference}",
                "max_length": {max_length}
            }}
        }}
        
        IMPORTANT: 
        - Each title must be exactly {max_length} characters or less
        - Provide 3 distinctly different approaches
        - Avoid repetition with recent context
        - Maintain the core meaning while enhancing clarity and impact
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.8
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
                enhanced_titles = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                # Fallback to basic structure
                enhanced_titles = {
                    "original_title": original_title,
                    "enhanced_titles": [
                        {
                            "title": f"Enhanced: {original_title[:max_length-10]}",
                            "style": "professional",
                            "approach": "Direct enhancement with clarity",
                            "character_count": len(f"Enhanced: {original_title[:max_length-10]}")
                        },
                        {
                            "title": f"Updated: {original_title[:max_length-9]}",
                            "style": "concise",
                            "approach": "Brief and to-the-point",
                            "character_count": len(f"Updated: {original_title[:max_length-9]}")
                        },
                        {
                            "title": f"Improved: {original_title[:max_length-10]}",
                            "style": "descriptive",
                            "approach": "More detailed and informative",
                            "character_count": len(f"Improved: {original_title[:max_length-10]}")
                        }
                    ],
                    "metadata": {
                        "context_type": context_type,
                        "domain": domain,
                        "style_preference": style_preference,
                        "max_length": max_length
                    }
                }
            
            # Add to database context
            enhanced_title_list = [title["title"] for title in enhanced_titles["enhanced_titles"]]
            self.add_title_to_context(
                original_title=original_title,
                enhanced_titles=enhanced_title_list,
                context_type=context_type,
                domain=domain
            )
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return enhanced_titles
            
        except Exception as e:
            logger.error(f"Error enhancing title: {e}")
            raise Exception(f"Failed to enhance title: {str(e)}")
    
    def regenerate_titles(self, original_title: str, context_type: str = "general", 
                         domain: str = "general", max_length: int = 50, 
                         style_preference: str = "professional", 
                         exclude_previous: bool = True) -> Dict:
        """Regenerate title suggestions with context awareness to avoid repetition"""
        
        # Get context to understand what was previously generated
        context_summary = self.get_title_context_summary(domain)
        
        prompt = f"""
        Create THREE NEW and DIFFERENT enhanced title suggestions for the given original title.
        IMPORTANT: These should be COMPLETELY DIFFERENT from any previous suggestions.
        
        PREVIOUS CONTEXT (AVOID THESE APPROACHES):
        {context_summary}
        
        INPUT INFORMATION:
        - Original Title: "{original_title}"
        - Context Type: {context_type}
        - Domain: {domain}
        - Style Preference: {style_preference}
        - Maximum Length: {max_length} characters
        - Exclude Previous: {exclude_previous}
        
        REQUIREMENTS:
        1. CREATE 3 COMPLETELY NEW APPROACHES: Different from any previous suggestions
        2. CHARACTER LIMIT: Maximum {max_length} characters per title
        3. AVOID REPETITION: Do not use similar approaches from previous context
        4. STYLE VARIETY: Provide different stylistic approaches
        5. FRESH PERSPECTIVE: Offer new angles and interpretations
        
        Return the response as JSON with the following structure:
        {{
            "original_title": "{original_title}",
            "enhanced_titles": [
                {{
                    "title": "First new enhanced title (max {max_length} chars)",
                    "style": "creative",
                    "approach": "New approach description",
                    "character_count": 25,
                    "uniqueness": "What makes this different from previous suggestions"
                }},
                {{
                    "title": "Second new enhanced title (max {max_length} chars)",
                    "style": "technical",
                    "approach": "New approach description", 
                    "character_count": 30,
                    "uniqueness": "What makes this different from previous suggestions"
                }},
                {{
                    "title": "Third new enhanced title (max {max_length} chars)",
                    "style": "attention-grabbing",
                    "approach": "New approach description",
                    "character_count": 20,
                    "uniqueness": "What makes this different from previous suggestions"
                }}
            ],
            "metadata": {{
                "context_type": "{context_type}",
                "domain": "{domain}",
                "style_preference": "{style_preference}",
                "max_length": {max_length},
                "regeneration": true
            }}
        }}
        
        IMPORTANT: 
        - Each title must be exactly {max_length} characters or less
        - Provide 3 completely new approaches
        - Avoid any similarity to previous suggestions
        - Maintain the core meaning while offering fresh perspectives
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.9
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
                enhanced_titles = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                # Fallback to basic structure with different approaches
                enhanced_titles = {
                    "original_title": original_title,
                    "enhanced_titles": [
                        {
                            "title": f"New: {original_title[:max_length-5]}",
                            "style": "creative",
                            "approach": "Fresh creative approach",
                            "character_count": len(f"New: {original_title[:max_length-5]}"),
                            "uniqueness": "Uses 'New' prefix for freshness"
                        },
                        {
                            "title": f"Revamped: {original_title[:max_length-9]}",
                            "style": "attention-grabbing",
                            "approach": "Bold and compelling",
                            "character_count": len(f"Revamped: {original_title[:max_length-9]}"),
                            "uniqueness": "Uses 'Revamped' for impact"
                        },
                        {
                            "title": f"Optimized: {original_title[:max_length-11]}",
                            "style": "technical",
                            "approach": "Technical optimization focus",
                            "character_count": len(f"Optimized: {original_title[:max_length-11]}"),
                            "uniqueness": "Uses 'Optimized' for technical appeal"
                        }
                    ],
                    "metadata": {
                        "context_type": context_type,
                        "domain": domain,
                        "style_preference": style_preference,
                        "max_length": max_length,
                        "regeneration": True
                    }
                }
            
            # Add to database context
            enhanced_title_list = [title["title"] for title in enhanced_titles["enhanced_titles"]]
            self.add_title_to_context(
                original_title=original_title,
                enhanced_titles=enhanced_title_list,
                context_type=context_type,
                domain=domain
            )
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return enhanced_titles
            
        except Exception as e:
            logger.error(f"Error regenerating titles: {e}")
            raise Exception(f"Failed to regenerate titles: {str(e)}")
    
    def get_title_suggestions_by_domain(self, domain: str) -> Dict:
        """Get title enhancement suggestions based on domain"""
        
        prompt = f"""
        Provide focused suggestions for creating effective titles in the {domain} domain.
        
        Return the response as JSON with the following structure:
        {{
            "domain": "{domain}",
            "title_patterns": [
                "First effective title pattern for {domain}",
                "Second title structure approach",
                "Third title format recommendation"
            ],
            "style_recommendations": [
                "First style recommendation for {domain} titles",
                "Second tone suggestion",
                "Third approach recommendation"
            ],
            "common_elements": [
                "First common element in {domain} titles",
                "Second typical component",
                "Third standard feature"
            ],
            "avoidance_tips": [
                "First thing to avoid in {domain} titles",
                "Second common mistake",
                "Third pitfall to watch out for"
            ],
            "domain_specific_notes": "Tailored recommendations for the {domain} domain"
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
            logger.error(f"Error getting title suggestions by domain: {e}")
            raise Exception(f"Failed to get title suggestions: {str(e)}")
    
    def clear_title_context(self, domain: str = None):
        """Clear the title context history from database"""
        try:
            success = db_manager.clear_title_context(domain)
            if success:
                return {"message": f"Title context history cleared successfully for domain: {domain or 'all'}"}
            else:
                return {"message": "Failed to clear title context history"}
        except Exception as e:
            logger.error(f"Error clearing title context: {e}")
            return {"message": f"Error clearing title context: {str(e)}"}

# Initialize title enhancer
title_enhancer = TitleEnhancer()

class MessageBodyEnhancer:
    """Enhanced message body generation with context management and unique suggestions"""
    
    def __init__(self):
        # Test database connection on initialization
        if not db_manager.test_connection():
            logger.warning("Database connection failed. Context will be limited.")
    
    def add_message_to_context(self, original_message: str, enhanced_messages: List[str], 
                             context_type: str = "message_enhancement", 
                             domain: str = "general"):
        """Add enhanced messages to database context for better continuity"""
        try:
            success = db_manager.add_message_context(
                original_message=original_message,
                enhanced_messages=enhanced_messages,
                context_type=context_type,
                domain=domain
            )
            if success:
                logger.info(f"Added message enhancement to context: {original_message[:30]}...")
            else:
                logger.error("Failed to add message enhancement to context")
        except Exception as e:
            logger.error(f"Error adding message to context: {e}")
    
    def get_message_context_summary(self, domain: str = "general") -> str:
        """Get a summary of recent message enhancements for context from database"""
        try:
            return db_manager.get_message_context_summary(domain=domain, limit=3)
        except Exception as e:
            logger.error(f"Error getting message context summary: {e}")
            return "No previous message enhancements."
    
    def enhance_message(self, original_message: str, context_type: str = "general", 
                       domain: str = "general", max_length: int = 500, 
                       style_preference: str = "professional", 
                       tone: str = "professional") -> Dict:
        """Create enhanced message suggestions with context awareness"""
        
        # Validate context type
        valid_context_types = [
            "general", "academic", "business", "creative", "technical", 
            "marketing", "educational", "professional", "casual", "formal"
        ]
        if context_type not in valid_context_types:
            context_type = "general"
        
        # Validate style preference
        valid_styles = [
            "professional", "creative", "concise", "descriptive", 
            "attention-grabbing", "formal", "casual", "technical", "persuasive"
        ]
        if style_preference not in valid_styles:
            style_preference = "professional"
        
        # Validate tone
        valid_tones = [
            "professional", "friendly", "formal", "casual", "enthusiastic",
            "serious", "encouraging", "informative", "persuasive", "neutral"
        ]
        if tone not in valid_tones:
            tone = "professional"
        
        # Build context-aware prompt
        context_summary = self.get_message_context_summary(domain)
        
        prompt = f"""
        Create THREE UNIQUE and DISTINCTLY DIFFERENT enhanced message suggestions for the given original message.
        
        CONTEXT:
        {context_summary}
        
        INPUT INFORMATION:
        - Original Message: "{original_message}"
        - Context Type: {context_type}
        - Domain: {domain}
        - Style Preference: {style_preference}
        - Tone: {tone}
        - Maximum Length: {max_length} characters
        
        REQUIREMENTS:
        1. CREATE 3 COMPLETELY DIFFERENT APPROACHES: Each suggestion should have a unique style, tone, and approach
        2. CHARACTER LIMIT: Maximum {max_length} characters per message - USE MOST OF THE AVAILABLE CHARACTERS
        3. CONTEXT AWARENESS: Consider recent message enhancements to avoid repetition and maintain consistency
        4. STYLE VARIETY: Provide different stylistic approaches while maintaining the core message
        5. DOMAIN APPROPRIATENESS: Ensure messages are suitable for the specified domain
        6. TONE CONSISTENCY: Maintain the specified tone throughout each message
        7. DETAILED CONTENT: Make messages comprehensive and informative, not brief or minimal
        8. ENGAGING LANGUAGE: Use compelling and descriptive language to make messages more impactful
        
        STYLE GUIDELINES:
        - Professional: Clear, structured, business-appropriate language with detailed explanations
        - Creative: Engaging, imaginative, attention-grabbing content with vivid descriptions
        - Concise: Direct, to-the-point communication while still being comprehensive
        - Descriptive: Detailed, informative, comprehensive explanations with rich context
        - Attention-grabbing: Bold, compelling, memorable content with strong emotional appeal
        - Formal: Structured, academic, official language with thorough explanations
        - Casual: Friendly, approachable, conversational tone with warm details
        - Technical: Precise, specialized, industry-specific language with detailed specifications
        - Persuasive: Compelling, convincing, action-oriented content with strong arguments
        
        TONE GUIDELINES:
        - Professional: Respectful, authoritative, business-like with comprehensive details
        - Friendly: Warm, approachable, personable with encouraging details
        - Formal: Official, structured, respectful with thorough explanations
        - Casual: Relaxed, informal, conversational with engaging details
        - Enthusiastic: Energetic, positive, excited with motivating details
        - Serious: Grave, important, weighty with comprehensive information
        - Encouraging: Supportive, motivating, uplifting with inspiring details
        - Informative: Educational, explanatory, clear with detailed information
        - Persuasive: Convincing, compelling, action-oriented with strong reasoning
        - Neutral: Balanced, objective, unbiased with comprehensive details
        
        MESSAGE ENHANCEMENT GUIDELINES:
        - EXPAND on the original message with additional relevant details
        - ADD context, explanations, or background information where appropriate
        - INCLUDE specific details, dates, times, locations, or other relevant information
        - USE engaging language that captures attention and maintains interest
        - PROVIDE comprehensive information that answers potential questions
        - MAKE messages more informative and valuable to the reader
        
        Return the response as JSON with the following structure:
        {{
            "original_message": "{original_message}",
            "enhanced_messages": [
                {{
                    "message": "First detailed enhanced message suggestion (use {max_length-50} to {max_length} chars)",
                    "style": "professional",
                    "tone": "professional",
                    "approach": "Brief description of the approach used",
                    "character_count": 150
                }},
                {{
                    "message": "Second detailed enhanced message suggestion (use {max_length-50} to {max_length} chars)",
                    "style": "creative", 
                    "tone": "friendly",
                    "approach": "Brief description of the approach used",
                    "character_count": 180
                }},
                {{
                    "message": "Third detailed enhanced message suggestion (use {max_length-50} to {max_length} chars)",
                    "style": "descriptive",
                    "tone": "informative",
                    "approach": "Brief description of the approach used", 
                    "character_count": 200
                }}
            ],
            "metadata": {{
                "context_type": "{context_type}",
                "domain": "{domain}",
                "style_preference": "{style_preference}",
                "tone": "{tone}",
                "max_length": {max_length}
            }}
        }}
        
        IMPORTANT: 
        - Each message should use 80-95% of the available {max_length} characters
        - Provide 3 distinctly different approaches with detailed content
        - Avoid repetition with recent context
        - Maintain the core meaning while enhancing clarity and impact
        - Ensure each message is complete, coherent, and comprehensive
        - Make messages more detailed and informative than the original
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.8
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
                enhanced_messages = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                # Fallback to basic structure
                enhanced_messages = {
                    "original_message": original_message,
                    "enhanced_messages": [
                        {
                            "message": f"Enhanced message: {original_message[:max_length-20]}",
                            "style": "professional",
                            "tone": "professional",
                            "approach": "Direct enhancement with clarity",
                            "character_count": len(f"Enhanced message: {original_message[:max_length-20]}")
                        },
                        {
                            "message": f"Updated communication: {original_message[:max_length-25]}",
                            "style": "concise",
                            "tone": "informative",
                            "approach": "Brief and to-the-point",
                            "character_count": len(f"Updated communication: {original_message[:max_length-25]}")
                        },
                        {
                            "message": f"Improved message: {original_message[:max_length-20]}",
                            "style": "descriptive",
                            "tone": "friendly",
                            "approach": "More detailed and informative",
                            "character_count": len(f"Improved message: {original_message[:max_length-20]}")
                        }
                    ],
                    "metadata": {
                        "context_type": context_type,
                        "domain": domain,
                        "style_preference": style_preference,
                        "tone": tone,
                        "max_length": max_length
                    }
                }
            
            # Add to database context
            enhanced_message_list = [msg["message"] for msg in enhanced_messages["enhanced_messages"]]
            self.add_message_to_context(
                original_message=original_message,
                enhanced_messages=enhanced_message_list,
                context_type=context_type,
                domain=domain
            )
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return enhanced_messages
            
        except Exception as e:
            logger.error(f"Error enhancing message: {e}")
            raise Exception(f"Failed to enhance message: {str(e)}")
    
    def regenerate_messages(self, original_message: str, context_type: str = "general", 
                           domain: str = "general", max_length: int = 500, 
                           style_preference: str = "professional", 
                           tone: str = "professional",
                           exclude_previous: bool = True) -> Dict:
        """Regenerate message suggestions with context awareness to avoid repetition"""
        
        # Get context to understand what was previously generated
        context_summary = self.get_message_context_summary(domain)
        
        prompt = f"""
        Create THREE NEW and DIFFERENT enhanced message suggestions for the given original message.
        IMPORTANT: These should be COMPLETELY DIFFERENT from any previous suggestions.
        
        PREVIOUS CONTEXT (AVOID THESE APPROACHES):
        {context_summary}
        
        INPUT INFORMATION:
        - Original Message: "{original_message}"
        - Context Type: {context_type}
        - Domain: {domain}
        - Style Preference: {style_preference}
        - Tone: {tone}
        - Maximum Length: {max_length} characters
        - Exclude Previous: {exclude_previous}
        
        REQUIREMENTS:
        1. CREATE 3 COMPLETELY NEW APPROACHES: Different from any previous suggestions
        2. CHARACTER LIMIT: Maximum {max_length} characters per message - USE MOST OF THE AVAILABLE CHARACTERS
        3. AVOID REPETITION: Do not use similar approaches from previous context
        4. STYLE VARIETY: Provide different stylistic approaches
        5. FRESH PERSPECTIVE: Offer new angles and interpretations
        6. TONE CONSISTENCY: Maintain the specified tone throughout
        7. DETAILED CONTENT: Make messages comprehensive and informative, not brief or minimal
        8. ENGAGING LANGUAGE: Use compelling and descriptive language to make messages more impactful
        
        MESSAGE ENHANCEMENT GUIDELINES:
        - EXPAND on the original message with additional relevant details
        - ADD context, explanations, or background information where appropriate
        - INCLUDE specific details, dates, times, locations, or other relevant information
        - USE engaging language that captures attention and maintains interest
        - PROVIDE comprehensive information that answers potential questions
        - MAKE messages more informative and valuable to the reader
        
        Return the response as JSON with the following structure:
        {{
            "original_message": "{original_message}",
            "enhanced_messages": [
                {{
                    "message": "First new detailed enhanced message (use {max_length-50} to {max_length} chars)",
                    "style": "creative",
                    "tone": "enthusiastic",
                    "approach": "New approach description",
                    "character_count": 150,
                    "uniqueness": "What makes this different from previous suggestions"
                }},
                {{
                    "message": "Second new detailed enhanced message (use {max_length-50} to {max_length} chars)",
                    "style": "technical",
                    "tone": "informative",
                    "approach": "New approach description", 
                    "character_count": 180,
                    "uniqueness": "What makes this different from previous suggestions"
                }},
                {{
                    "message": "Third new detailed enhanced message (use {max_length-50} to {max_length} chars)",
                    "style": "persuasive",
                    "tone": "encouraging",
                    "approach": "New approach description",
                    "character_count": 200,
                    "uniqueness": "What makes this different from previous suggestions"
                }}
            ],
            "metadata": {{
                "context_type": "{context_type}",
                "domain": "{domain}",
                "style_preference": "{style_preference}",
                "tone": "{tone}",
                "max_length": {max_length},
                "regeneration": true
            }}
        }}
        
        IMPORTANT: 
        - Each message should use 80-95% of the available {max_length} characters
        - Provide 3 completely new approaches with detailed content
        - Avoid any similarity to previous suggestions
        - Maintain the core meaning while offering fresh perspectives
        - Ensure each message is complete, coherent, and comprehensive
        - Make messages more detailed and informative than the original
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                temperature=0.9
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
                enhanced_messages = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                # Fallback to basic structure with different approaches
                enhanced_messages = {
                    "original_message": original_message,
                    "enhanced_messages": [
                        {
                            "message": f"New approach: {original_message[:max_length-15]}",
                            "style": "creative",
                            "tone": "enthusiastic",
                            "approach": "Fresh creative approach",
                            "character_count": len(f"New approach: {original_message[:max_length-15]}"),
                            "uniqueness": "Uses 'New approach' prefix for freshness"
                        },
                        {
                            "message": f"Revamped communication: {original_message[:max_length-25]}",
                            "style": "persuasive",
                            "tone": "encouraging",
                            "approach": "Bold and compelling",
                            "character_count": len(f"Revamped communication: {original_message[:max_length-25]}"),
                            "uniqueness": "Uses 'Revamped communication' for impact"
                        },
                        {
                            "message": f"Optimized message: {original_message[:max_length-20]}",
                            "style": "technical",
                            "tone": "informative",
                            "approach": "Technical optimization focus",
                            "character_count": len(f"Optimized message: {original_message[:max_length-20]}"),
                            "uniqueness": "Uses 'Optimized message' for technical appeal"
                        }
                    ],
                    "metadata": {
                        "context_type": context_type,
                        "domain": domain,
                        "style_preference": style_preference,
                        "tone": tone,
                        "max_length": max_length,
                        "regeneration": True
                    }
                }
            
            # Add to database context
            enhanced_message_list = [msg["message"] for msg in enhanced_messages["enhanced_messages"]]
            self.add_message_to_context(
                original_message=original_message,
                enhanced_messages=enhanced_message_list,
                context_type=context_type,
                domain=domain
            )
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            token_tracker.add_tokens(tokens_used)
            
            return enhanced_messages
            
        except Exception as e:
            logger.error(f"Error regenerating messages: {e}")
            raise Exception(f"Failed to regenerate messages: {str(e)}")
    
    def get_message_suggestions_by_domain(self, domain: str) -> Dict:
        """Get message enhancement suggestions based on domain"""
        
        prompt = f"""
        Provide focused suggestions for creating effective messages in the {domain} domain.
        
        Return the response as JSON with the following structure:
        {{
            "domain": "{domain}",
            "message_patterns": [
                "First effective message pattern for {domain}",
                "Second message structure approach",
                "Third message format recommendation"
            ],
            "style_recommendations": [
                "First style recommendation for {domain} messages",
                "Second tone suggestion",
                "Third approach recommendation"
            ],
            "common_elements": [
                "First common element in {domain} messages",
                "Second typical component",
                "Third standard feature"
            ],
            "avoidance_tips": [
                "First thing to avoid in {domain} messages",
                "Second common mistake",
                "Third pitfall to watch out for"
            ],
            "domain_specific_notes": "Tailored recommendations for the {domain} domain"
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
            logger.error(f"Error getting message suggestions by domain: {e}")
            raise Exception(f"Failed to get message suggestions: {str(e)}")
    
    def clear_message_context(self, domain: str = None):
        """Clear the message context history from database"""
        try:
            success = db_manager.clear_message_context(domain)
            if success:
                return {"message": f"Message context history cleared successfully for domain: {domain or 'all'}"}
            else:
                return {"message": "Failed to clear message context history"}
        except Exception as e:
            logger.error(f"Error clearing message context: {e}")
            return {"message": f"Error clearing message context: {str(e)}"}

# Initialize message body enhancer
message_body_enhancer = MessageBodyEnhancer()

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
        
        # Check token limit for grading
        estimated_tokens = len(questions) * 50  # Rough estimate per question
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        
        # Check token limit for assessment creation
        estimated_tokens = 500  # Rough estimate for assessment generation
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        # Check token limit for context retrieval
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        # Check token limit for context clearing
        estimated_tokens = 50  # Rough estimate for context clearing
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        
        # Check token limit for context retrieval by type
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        
        # Check token limit for context retrieval by audience
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        # Check token limit for stats retrieval
        estimated_tokens = 50  # Rough estimate for stats retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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
        # Check token limit for deletion
        estimated_tokens = 50  # Rough estimate for deletion
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
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

@app.route('/api/title/enhance', methods=['POST'])
def enhance_title():
    """Enhance a title with 3 unique suggestions"""
    try:
        data = request.get_json()
        original_title = data.get('original_title')
        context_type = data.get('context_type', 'general')
        domain = data.get('domain', 'general')
        max_length = data.get('max_length', 50)
        style_preference = data.get('style_preference', 'professional')
        
        # Validate required parameters
        if not original_title:
            return jsonify({
                "error": "Missing required parameter: original_title"
            }), 400
        
        # Validate max_length
        if max_length < 10 or max_length > 200:
            return jsonify({
                "error": "max_length must be between 10 and 200 characters"
            }), 400
        
        # Check token limit
        estimated_tokens = 400  # Rough estimate for title enhancement
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        enhanced_titles = title_enhancer.enhance_title(
            original_title=original_title,
            context_type=context_type,
            domain=domain,
            max_length=max_length,
            style_preference=style_preference
        )
        
        # Track LLM usage for title enhancement
        # Estimate tokens used (rough calculation based on input + output)
        estimated_tokens = len(original_title) + sum(len(title["title"]) for title in enhanced_titles["enhanced_titles"]) + 200  # Add buffer for prompt
        estimated_cost = (estimated_tokens / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
        track_llm_text_usage(estimated_tokens, estimated_cost, "title_enhancement")
        
        # Extract just the titles as a simple array
        title_array = [title["title"] for title in enhanced_titles["enhanced_titles"]]
        
        # Convert to array of arrays format
        titles_object = {}
        for i, title in enumerate(title_array, 1):
            titles_object[f"title{i}"] = title
        
        return jsonify({
            "success": True,
            "titles": titles_object
        })
        
    except Exception as e:
        logger.error(f"Error in enhance_title: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/regenerate', methods=['POST'])
def regenerate_titles():
    """Regenerate title suggestions with context awareness"""
    try:
        data = request.get_json()
        original_title = data.get('original_title')
        context_type = data.get('context_type', 'general')
        domain = data.get('domain', 'general')
        max_length = data.get('max_length', 50)
        style_preference = data.get('style_preference', 'professional')
        exclude_previous = data.get('exclude_previous', True)
        
        # Validate required parameters
        if not original_title:
            return jsonify({
                "error": "Missing required parameter: original_title"
            }), 400
        
        # Validate max_length
        if max_length < 10 or max_length > 200:
            return jsonify({
                "error": "max_length must be between 10 and 200 characters"
            }), 400
        
        # Check token limit
        estimated_tokens = 450  # Rough estimate for title regeneration
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        enhanced_titles = title_enhancer.regenerate_titles(
            original_title=original_title,
            context_type=context_type,
            domain=domain,
            max_length=max_length,
            style_preference=style_preference,
            exclude_previous=exclude_previous
        )
        
        # Track LLM usage for title regeneration
        # Estimate tokens used (rough calculation based on input + output)
        estimated_tokens = len(original_title) + sum(len(title["title"]) for title in enhanced_titles["enhanced_titles"]) + 250  # Add buffer for regeneration prompt
        estimated_cost = (estimated_tokens / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
        track_llm_text_usage(estimated_tokens, estimated_cost, "title_regeneration")
        
        # Extract just the titles as a simple array
        title_array = [title["title"] for title in enhanced_titles["enhanced_titles"]]
        
        # Convert to array of arrays format
        titles_object = {}
        for i, title in enumerate(title_array, 1):
            titles_object[f"title{i}"] = title
        
        return jsonify({
            "success": True,
            "titles": titles_object
        })
        
    except Exception as e:
        logger.error(f"Error in regenerate_titles: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/suggestions/<domain>', methods=['GET'])
def get_title_suggestions_by_domain(domain: str):
    """Get title enhancement suggestions for a specific domain"""
    try:
        # Check token limit
        estimated_tokens = 300  # Rough estimate for domain suggestions
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        suggestions = title_enhancer.get_title_suggestions_by_domain(domain)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "metadata": {
                "domain": domain,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_title_suggestions_by_domain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/context', methods=['GET'])
def get_title_context():
    """Get current title enhancement context history from database"""
    try:
        domain = request.args.get('domain', 'general')
        limit = request.args.get('limit', 5, type=int)
        
        # Check token limit for context retrieval
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        context_summary = title_enhancer.get_title_context_summary(domain)
        recent_titles = db_manager.get_recent_title_enhancements(domain, limit)
        stats = db_manager.get_title_enhancement_stats(domain)
        
        return jsonify({
            "success": True,
            "context_summary": context_summary,
            "domain": domain,
            "recent_enhancements": recent_titles,
            "stats": stats,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_title_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/context/clear', methods=['POST'])
def clear_title_context():
    """Clear the title enhancement context history from database"""
    try:
        data = request.get_json() or {}
        domain = data.get('domain')
        
        # Check token limit for context clearing
        estimated_tokens = 50  # Rough estimate for context clearing
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        result = title_enhancer.clear_title_context(domain)
        
        return jsonify({
            "success": True,
            "result": result,
            "metadata": {
                "cleared_at": datetime.now().isoformat(),
                "domain": domain,
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in clear_title_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/context/by-type/<context_type>', methods=['GET'])
def get_title_enhancements_by_type(context_type: str):
    """Get title enhancements by context type from database"""
    try:
        limit = request.args.get('limit', 5, type=int)
        domain = request.args.get('domain', 'general')
        
        # Check token limit for context retrieval by type
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        enhancements = db_manager.get_title_enhancements_by_type(context_type, domain, limit)
        
        return jsonify({
            "success": True,
            "context_type": context_type,
            "domain": domain,
            "enhancements": enhancements,
            "count": len(enhancements),
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_title_enhancements_by_type: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/context/stats', methods=['GET'])
def get_title_enhancement_stats():
    """Get title enhancement statistics from database"""
    try:
        domain = request.args.get('domain', 'general')
        
        # Check token limit for stats retrieval
        estimated_tokens = 50  # Rough estimate for stats retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        stats = db_manager.get_title_enhancement_stats(domain)
        
        return jsonify({
            "success": True,
            "domain": domain,
            "stats": stats,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_title_enhancement_stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/title/context/delete/<int:enhancement_id>', methods=['DELETE'])
def delete_title_enhancement(enhancement_id: int):
    """Delete specific title enhancement from database"""
    try:
        # Check token limit for deletion
        estimated_tokens = 50  # Rough estimate for deletion
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        success = db_manager.delete_title_enhancement(enhancement_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Title enhancement {enhancement_id} deleted successfully",
                "metadata": {
                    "deleted_at": datetime.now().isoformat(),
                    "database_connected": db_manager.test_connection()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to delete title enhancement {enhancement_id}",
                "metadata": {
                    "attempted_at": datetime.now().isoformat(),
                    "database_connected": db_manager.test_connection()
                }
            }), 404
        
    except Exception as e:
        logger.error(f"Error in delete_title_enhancement: {e}")
        return jsonify({"error": str(e)}), 500

# Message Body Enhancement API Routes

@app.route('/api/message/enhance', methods=['POST'])
def enhance_message():
    """Enhance a message with 3 unique suggestions"""
    try:
        data = request.get_json()
        original_message = data.get('original_message')
        context_type = data.get('context_type', 'general')
        domain = data.get('domain', 'general')
        max_length = data.get('max_length', 500)
        style_preference = data.get('style_preference', 'professional')
        tone = data.get('tone', 'professional')
        
        # Validate required parameters
        if not original_message:
            return jsonify({
                "error": "Missing required parameter: original_message"
            }), 400
        
        # Validate max_length
        if max_length < 50 or max_length > 2000:
            return jsonify({
                "error": "max_length must be between 50 and 2000 characters"
            }), 400
        
        # Check token limit
        estimated_tokens = 600  # Rough estimate for message enhancement
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        enhanced_messages = message_body_enhancer.enhance_message(
            original_message=original_message,
            context_type=context_type,
            domain=domain,
            max_length=max_length,
            style_preference=style_preference,
            tone=tone
        )
        
        # Track LLM usage for message enhancement
        # Estimate tokens used (rough calculation based on input + output)
        estimated_tokens = len(original_message) + sum(len(msg["message"]) for msg in enhanced_messages["enhanced_messages"]) + 300  # Add buffer for prompt
        estimated_cost = (estimated_tokens / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
        track_llm_text_usage(estimated_tokens, estimated_cost, "message_enhancement")
        
        # Extract just the messages as a simple array
        message_array = [msg["message"] for msg in enhanced_messages["enhanced_messages"]]
        
        # Convert to array of arrays format
        messages_object = {}
        for i, message in enumerate(message_array, 1):
            messages_object[f"message{i}"] = message
        
        return jsonify({
            "success": True,
            "messages": messages_object
        })
        
    except Exception as e:
        logger.error(f"Error in enhance_message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/regenerate', methods=['POST'])
def regenerate_messages():
    """Regenerate message suggestions with context awareness"""
    try:
        data = request.get_json()
        original_message = data.get('original_message')
        context_type = data.get('context_type', 'general')
        domain = data.get('domain', 'general')
        max_length = data.get('max_length', 500)
        style_preference = data.get('style_preference', 'professional')
        tone = data.get('tone', 'professional')
        exclude_previous = data.get('exclude_previous', True)
        
        # Validate required parameters
        if not original_message:
            return jsonify({
                "error": "Missing required parameter: original_message"
            }), 400
        
        # Validate max_length
        if max_length < 50 or max_length > 2000:
            return jsonify({
                "error": "max_length must be between 50 and 2000 characters"
            }), 400
        
        # Check token limit
        estimated_tokens = 650  # Rough estimate for message regeneration
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        enhanced_messages = message_body_enhancer.regenerate_messages(
            original_message=original_message,
            context_type=context_type,
            domain=domain,
            max_length=max_length,
            style_preference=style_preference,
            tone=tone,
            exclude_previous=exclude_previous
        )
        
        # Track LLM usage for message regeneration
        # Estimate tokens used (rough calculation based on input + output)
        estimated_tokens = len(original_message) + sum(len(msg["message"]) for msg in enhanced_messages["enhanced_messages"]) + 350  # Add buffer for regeneration prompt
        estimated_cost = (estimated_tokens / 1000) * 0.002  # Rough estimate: $0.002 per 1K tokens
        track_llm_text_usage(estimated_tokens, estimated_cost, "message_regeneration")
        
        # Extract just the messages as a simple array
        message_array = [msg["message"] for msg in enhanced_messages["enhanced_messages"]]
        
        # Convert to array of arrays format
        messages_object = {}
        for i, message in enumerate(message_array, 1):
            messages_object[f"message{i}"] = message
        
        return jsonify({
            "success": True,
            "messages": messages_object
        })
        
    except Exception as e:
        logger.error(f"Error in regenerate_messages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/suggestions/<domain>', methods=['GET'])
def get_message_suggestions_by_domain(domain: str):
    """Get message enhancement suggestions for a specific domain"""
    try:
        # Check token limit
        estimated_tokens = 300  # Rough estimate for domain suggestions
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        suggestions = message_body_enhancer.get_message_suggestions_by_domain(domain)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "metadata": {
                "domain": domain,
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_message_suggestions_by_domain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/context', methods=['GET'])
def get_message_context():
    """Get current message enhancement context history from database"""
    try:
        domain = request.args.get('domain', 'general')
        limit = request.args.get('limit', 5, type=int)
        
        # Check token limit for context retrieval
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        context_summary = message_body_enhancer.get_message_context_summary(domain)
        recent_messages = db_manager.get_recent_message_enhancements(domain, limit)
        stats = db_manager.get_message_enhancement_stats(domain)
        
        return jsonify({
            "success": True,
            "context_summary": context_summary,
            "domain": domain,
            "recent_enhancements": recent_messages,
            "stats": stats,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_message_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/context/clear', methods=['POST'])
def clear_message_context():
    """Clear the message enhancement context history from database"""
    try:
        data = request.get_json() or {}
        domain = data.get('domain')
        
        # Check token limit for context clearing
        estimated_tokens = 50  # Rough estimate for context clearing
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        result = message_body_enhancer.clear_message_context(domain)
        
        return jsonify({
            "success": True,
            "result": result,
            "metadata": {
                "cleared_at": datetime.now().isoformat(),
                "domain": domain,
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in clear_message_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/context/by-type/<context_type>', methods=['GET'])
def get_message_enhancements_by_type(context_type: str):
    """Get message enhancements by context type from database"""
    try:
        limit = request.args.get('limit', 5, type=int)
        domain = request.args.get('domain', 'general')
        
        # Check token limit for context retrieval by type
        estimated_tokens = 100  # Rough estimate for context retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        enhancements = db_manager.get_message_enhancements_by_type(context_type, domain, limit)
        
        return jsonify({
            "success": True,
            "context_type": context_type,
            "domain": domain,
            "enhancements": enhancements,
            "count": len(enhancements),
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_message_enhancements_by_type: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/context/stats', methods=['GET'])
def get_message_enhancement_stats():
    """Get message enhancement statistics from database"""
    try:
        domain = request.args.get('domain', 'general')
        
        # Check token limit for stats retrieval
        estimated_tokens = 50  # Rough estimate for stats retrieval
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        stats = db_manager.get_message_enhancement_stats(domain)
        
        return jsonify({
            "success": True,
            "domain": domain,
            "stats": stats,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_message_enhancement_stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/message/context/delete/<int:enhancement_id>', methods=['DELETE'])
def delete_message_enhancement(enhancement_id: int):
    """Delete specific message enhancement from database"""
    try:
        # Check token limit for deletion
        estimated_tokens = 50  # Rough estimate for deletion
        if not token_tracker.check_token_limit(estimated_tokens):
            return jsonify({
                "error": "Daily token limit exceeded",
                "usage": token_tracker.get_daily_usage()
            }), 429
        
        success = db_manager.delete_message_enhancement(enhancement_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Message enhancement {enhancement_id} deleted successfully",
                "metadata": {
                    "deleted_at": datetime.now().isoformat(),
                    "database_connected": db_manager.test_connection()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to delete message enhancement {enhancement_id}",
                "metadata": {
                    "attempted_at": datetime.now().isoformat(),
                    "database_connected": db_manager.test_connection()
                }
            }), 404
        
    except Exception as e:
        logger.error(f"Error in delete_message_enhancement: {e}")
        return jsonify({"error": str(e)}), 500

# Database-based LLM Usage Tracking
# Using database_manager.py for all LLM usage tracking operations

def track_llm_text_usage(tokens_used: int, cost_usd: float, request_type: str = "text_generation"):
    """Track text generation usage in database"""
    try:
        success = db_manager.add_llm_text_usage(tokens_used, cost_usd, request_type)
        if success:
            logger.info(f"LLM text usage tracked: {tokens_used} tokens, ${cost_usd:.4f}")
        else:
            logger.error("Failed to track LLM text usage in database")
        return success
    except Exception as e:
        logger.error(f"Error tracking LLM text usage: {e}")
        return False

def track_llm_image_usage(cost_usd: float, request_type: str = "image_generation"):
    """Track image generation usage in database"""
    try:
        success = db_manager.add_llm_image_usage(cost_usd, request_type)
        if success:
            logger.info(f"LLM image usage tracked: ${cost_usd:.4f}")
        else:
            logger.error("Failed to track LLM image usage in database")
        return success
    except Exception as e:
        logger.error(f"Error tracking LLM image usage: {e}")
        return False

# LLM Usage Analytics API Endpoints

@app.route('/api/llm-usage', methods=['GET'])
def get_llm_usage_summary():
    """Get comprehensive LLM usage summary and analytics"""
    try:
        usage_summary = db_manager.get_llm_usage_summary()
        
        return jsonify({
            "success": True,
            "usage_summary": usage_summary,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "tracking_active": True,
                "database_connected": db_manager.test_connection()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting LLM usage summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/llm-usage/daily/<date>', methods=['GET'])
def get_daily_llm_usage(date: str):
    """Get detailed LLM usage for a specific date"""
    try:
        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        daily_usage = db_manager.get_llm_daily_usage(date)
        
        if not daily_usage:
            return jsonify({
                "success": False,
                "message": f"No usage data found for {date}",
                "date": date
            }), 404
        
        return jsonify({
            "success": True,
            "date": date,
            "daily_usage": daily_usage,
            "metadata": {
                "retrieved_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting daily LLM usage: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/llm-usage/monthly/<month>', methods=['GET'])
def get_monthly_llm_usage(month: str):
    """Get detailed LLM usage for a specific month"""
    try:
        # Validate month format
        try:
            datetime.strptime(month, '%Y-%m')
        except ValueError:
            return jsonify({"error": "Invalid month format. Use YYYY-MM"}), 400
        
        monthly_usage = db_manager.get_llm_monthly_usage(month)
        
        if not monthly_usage:
            return jsonify({
                "success": False,
                "message": f"No usage data found for {month}",
                "month": month
            }), 404
        
        return jsonify({
            "success": True,
            "month": month,
            "monthly_usage": monthly_usage,
            "metadata": {
                "retrieved_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting monthly LLM usage: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/llm-usage/trends', methods=['GET'])
def get_llm_usage_trends():
    """Get usage trends and analytics"""
    try:
        days = request.args.get('days', 30, type=int)
        
        # Validate days parameter
        if days < 7 or days > 90:
            return jsonify({"error": "Days parameter must be between 7 and 90"}), 400
        
        # Get trends from database
        usage_summary = db_manager.get_llm_usage_summary()
        trends = usage_summary.get("usage_trends", {})
        
        # Filter cost trends to requested days
        if trends.get("cost_trends"):
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            filtered_trends = [
                trend for trend in trends["cost_trends"] 
                if trend["date"] >= cutoff_date
            ]
            trends["cost_trends"] = filtered_trends
        
        return jsonify({
            "success": True,
            "trends": trends,
            "analysis_period_days": days,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "data_points": len(trends["cost_trends"])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting LLM usage trends: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/llm-usage/insights', methods=['GET'])
def get_llm_usage_insights():
    """Get AI-generated insights about usage patterns"""
    try:
        # Get insights from database
        usage_summary = db_manager.get_llm_usage_summary()
        insights = usage_summary.get("usage_insights", {})
        trends = usage_summary.get("usage_trends", {})
        
        additional_insights = {}
        
        if trends.get("daily_averages"):
            avg_cost = trends["daily_averages"]["avg_cost_per_day"]
            if avg_cost > 0.10:
                additional_insights["cost_alert"] = "High daily cost detected"
            elif avg_cost < 0.01:
                additional_insights["cost_alert"] = "Very low daily cost"
        
        if trends.get("peak_usage_days"):
            peak_day, peak_cost = trends["peak_usage_days"][0]
            additional_insights["peak_day"] = {
                "date": peak_day,
                "cost": peak_cost,
                "note": "Highest cost day"
            }
        
        return jsonify({
            "success": True,
            "insights": insights,
            "additional_insights": additional_insights,
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "insights_generated": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting LLM usage insights: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
