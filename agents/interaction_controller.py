"""
Interaction Controller Module
Generates clarification questions based on detected signals
to avoid unnecessary code refactoring.
"""


def generate_questions(signals: dict) -> list:
    """
    Generate clarification questions based on detected signals.
    
    Args:
        signals (dict): Output from signal_detector.detect_signals()
        
    Returns:
        list: List of clarification questions to ask the user
    """
    questions = []
    
    # Always ask these mandatory clarification questions
    mandatory_questions = [
        "What is the EXACT error message or behavior you see? (This is critical!)",
        "When exactly does the issue NOT work?",
        "Which device, OS, and browser are you using?",
        "What exact user interaction triggers the issue?"
    ]
    
    # Category A specific questions
    if signals["category_a"]:
        questions.append("Can you describe the specific conditions when it works vs. when it doesn't?")
        questions.append("Is there a pattern to when the issue occurs?")
        
    # Category B specific questions
    if signals["category_b"]:
        questions.append("Does this happen consistently on the same platform, or across multiple platforms?")
        questions.append("Have you tested this on different browsers or devices?")
        
    # Add mandatory questions
    questions.extend(mandatory_questions)
    
    return questions


def format_investigation_response(signals: dict, questions: list) -> str:
    """
    Format an investigation response with detected signals and questions.
    
    Args:
        signals (dict): Detected signals
        questions (list): Generated questions
        
    Returns:
        str: Formatted response for the user
    """
    response = "[INVESTIGATION] Investigation Mode Activated\n\n"
    
    if signals["detected_keywords"]:
        response += "I detected these signals in your description:\n"
        for keyword in signals["detected_keywords"]:
            response += f"  * '{keyword}'\n"
        response += "\n"
    
    response += "This suggests the core logic might be correct, and the issue could be related to:\n"
    
    if signals["category_a"]:
        response += "  * Specific triggering conditions or timing\n"
    if signals["category_b"]:
        response += "  * Platform or environment-specific behavior\n"
    
    response += "\n**Before analyzing the code, please help me understand:**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"{i}. {question}\n"
    
    response += "\nThis will help me provide a more accurate solution without unnecessary refactoring."
    
    return response
