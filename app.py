from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import openai
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
from dotenv import load_dotenv

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
openai.api_key = Config.OPENAI_API_KEY
if not openai.api_key or openai.api_key == 'your-openai-api-key-here':
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
            response = openai.ChatCompletion.create(
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
            response = openai.ChatCompletion.create(
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
        Grade the following essay based on the provided rubric.
        
        Essay Prompt: {essay_prompt['prompt']}
        Word Limit: {essay_prompt['word_limit']} words
        
        Rubric:
        - Content: {essay_prompt['rubric']['content']}
        - Organization: {essay_prompt['rubric']['organization']}
        - Grammar: {essay_prompt['rubric']['grammar']}
        - Creativity: {essay_prompt['rubric']['creativity']}
        
        Student Essay:
        {student_essay}
        
        Return the response as JSON with the following structure:
        {{
            "overall_score": 85,
            "word_count": 450,
            "detailed_scores": {{
                "content": {{
                    "score": 20,
                    "max_score": 25,
                    "feedback": "Detailed feedback on content"
                }},
                "organization": {{
                    "score": 18,
                    "max_score": 25,
                    "feedback": "Detailed feedback on organization"
                }},
                "grammar": {{
                    "score": 22,
                    "max_score": 25,
                    "feedback": "Detailed feedback on grammar"
                }},
                "creativity": {{
                    "score": 20,
                    "max_score": 25,
                    "feedback": "Detailed feedback on creativity"
                }}
            }},
            "overall_feedback": "Comprehensive feedback on the essay",
            "grade": "B+"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
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

# API Routes

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
