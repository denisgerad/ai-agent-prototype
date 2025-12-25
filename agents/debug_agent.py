"""
Debug Agent
Main agent for handling bug reports with intelligent signal detection
and investigation before suggesting code changes.
"""

from signal_detector import detect_signals, should_investigate
from interaction_controller import generate_questions, format_investigation_response
from prompt_builder import build_prompt, build_analysis_prompt_with_answers
from code_inspector import generate_inspection_for_conversation
from verification_generator import generate_verification_for_issue
from root_cause_analyzer import generate_root_cause_analysis
from fix_strategy_generator import generate_fix_strategies_for_issue
from confirmation_gate import generate_confirmation_for_issue


class DebugAgent:
    """
    Agent that intelligently handles debugging requests by:
    1. Detecting stability and environment signals
    2. Asking clarification questions when needed
    3. Building appropriate prompts with debugging instructions
    """
    
    def __init__(self):
        self.clarifications = {}
        self.investigation_started = False
        self.original_issue = None
    
    def handle(self, user_input: str, code_context: str = None, conversation_history: str = None) -> dict:
        """
        Process a bug report and determine appropriate action.
        
        Args:
            user_input (str): The user's bug report or issue description
            code_context (str): Optional code context to analyze
            conversation_history (str): Previous conversation context
            
        Returns:
            dict: Response containing mode and appropriate data
        """
        # If we're in an ongoing investigation and user is providing more info
        if self.investigation_started and conversation_history:
            # Extract error messages from user input and history
            error_keywords = ['error', 'message', 'console', 'exception', 'failed', 'undefined', 'null', 'token', 'cors', 'network', 'status']
            has_error_in_input = any(keyword in user_input.lower() for keyword in error_keywords)
            has_error_in_history = any(keyword in conversation_history.lower() for keyword in error_keywords)
            
            # Generate enhanced analysis components
            inspection_checklist = generate_inspection_for_conversation(user_input, conversation_history)
            verification_tests = generate_verification_for_issue(user_input, conversation_history)
            root_cause_scores = generate_root_cause_analysis(user_input, conversation_history)
            fix_strategies = generate_fix_strategies_for_issue(user_input, conversation_history)
            confirmation_gate = generate_confirmation_for_issue(user_input, conversation_history)
            
            # Build prompt with emphasis on error messages
            prompt = build_prompt(
                f"Original issue: {self.original_issue}\n\nFollow-up information: {user_input}",
                code_context,
                None
            )
            
            if has_error_in_input or has_error_in_history:
                prompt += "\n\n" + "="*80 + "\n"
                prompt += "🚨 CRITICAL ERROR MESSAGE DETECTED 🚨\n"
                prompt += "="*80 + "\n\n"
                prompt += "The user has provided a SPECIFIC ERROR MESSAGE. This changes everything!\n\n"
                prompt += "MANDATORY ANALYSIS STEPS:\n"
                prompt += "1. Extract the EXACT error message from above (quote it)\n"
                prompt += "2. Explain what THIS SPECIFIC error means on the mentioned platform\n"
                prompt += "3. List 3-5 known causes of THIS EXACT error\n"
                prompt += "4. Provide targeted debugging steps for THIS error only\n"
                prompt += "5. Suggest fixes specific to THIS error, not general improvements\n\n"
                prompt += "ABSOLUTELY FORBIDDEN:\n"
                prompt += "❌ Generic 'browser compatibility' advice\n"
                prompt += "❌ Suggestions about touch events, DOM manipulation without relating to the error\n"
                prompt += "❌ General best practices that don't address the error\n"
                prompt += "❌ Code refactoring suggestions unrelated to the error\n\n"
                prompt += "EXAMPLE OF GOOD ANALYSIS (for 'no token' error):\n"
                prompt += "- Diagnose: 'no token' means authentication token not found\n"
                prompt += "- Platform: iOS Safari has restricted localStorage in certain modes\n"
                prompt += "- Solution: Check localStorage availability, use alternative storage\n\n"
                prompt += "="*80 + "\n\n"
            
            prompt += f"Conversation history:\n{conversation_history}\n\n"
            prompt += "## Your Analysis Task:\n\n"
            
            if has_error_in_input or has_error_in_history:
                prompt += "Start your response by stating: 'Error message detected: [exact error]'\n"
                prompt += "Then follow the MANDATORY ANALYSIS STEPS above.\n"
                prompt += "Do NOT deviate into generic platform advice!\n"
            else:
                prompt += "Analyze the issue and provide targeted recommendations.\n"
            
            # Reset investigation flag since we're now analyzing
            self.investigation_started = False
            
            return {
                "mode": "ANALYSIS",
                "prompt": prompt,
                "signals": None,
                "inspection_checklist": inspection_checklist,
                "verification_tests": verification_tests,
                "root_cause_scores": root_cause_scores,
                "fix_strategies": fix_strategies,
                "confirmation_gate": confirmation_gate
            }
        
        # Detect signals in the user input
        signals = detect_signals(user_input)
        
        # If signals detected, enter investigation mode
        if should_investigate(signals):
            self.investigation_started = True
            self.original_issue = user_input
            questions = generate_questions(signals)
            formatted_response = format_investigation_response(signals, questions)
            
            return {
                "mode": "INVESTIGATION",
                "questions": questions,
                "formatted_response": formatted_response,
                "signals": signals
            }
        
        # No signals detected, proceed with analysis
        prompt = build_prompt(user_input, code_context, signals)
        
        return {
            "mode": "ANALYSIS",
            "prompt": prompt,
            "signals": signals
        }
    
    def handle_with_clarifications(self, user_input: str, code_context: str, 
                                   clarifications: dict) -> dict:
        """
        Process a bug report after clarification questions have been answered.
        
        Args:
            user_input (str): Original bug report
            code_context (str): Code to analyze
            clarifications (dict): Answers to clarification questions
            
        Returns:
            dict: Response with analysis prompt
        """
        prompt = build_analysis_prompt_with_answers(
            user_input, 
            code_context, 
            clarifications
        )
        
        return {
            "mode": "ANALYSIS",
            "prompt": prompt
        }
    
    def is_debug_query(self, user_input: str) -> bool:
        """
        Determine if user input is a debug/bug query.
        
        Args:
            user_input (str): User's input
            
        Returns:
            bool: True if this looks like a debugging request
        """
        debug_keywords = [
            "bug", "issue", "not working", "error", "broken", 
            "doesn't work", "won't work", "fails", "problem",
            "crash", "freeze", "stuck"
        ]
        
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in debug_keywords)
    
    def reset(self):
        """Reset the debug agent state."""
        self.clarifications = {}
        self.investigation_started = False
        self.original_issue = None
