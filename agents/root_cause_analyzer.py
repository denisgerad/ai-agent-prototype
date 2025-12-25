"""
Root Cause Analyzer with Confidence Scoring
Generates likelihood scores for different root causes
"""


def calculate_root_cause_confidence(user_input: str, conversation_history: str) -> dict:
    """
    Calculate confidence scores for different root causes.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        dict: Root causes with confidence scores
    """
    combined_text = (user_input + " " + conversation_history).lower()
    root_causes = {}
    
    # Token/Authentication issues
    if "token" in combined_text and ("ios" in combined_text or "safari" in combined_text):
        root_causes["iOS Safari storage restrictions (Private Mode, ITP, localStorage failures)"] = 0.65
        root_causes["Authorization header not sent with request"] = 0.25
        root_causes["Token expired or invalid"] = 0.10
    
    elif "token" in combined_text:
        root_causes["Authorization header missing from request"] = 0.40
        root_causes["Token storage mechanism failure"] = 0.30
        root_causes["Token expired or invalid"] = 0.20
        root_causes["API endpoint requires different auth method"] = 0.10
    
    # iOS/Safari specific issues
    if ("ios" in combined_text or "safari" in combined_text) and "android" in combined_text:
        if "works" in combined_text and "android" in combined_text:
            root_causes["iOS Safari storage restrictions (Private Mode, ITP, localStorage)"] = 0.50
            root_causes["Safari event handling differences (click vs touchstart on delete)"] = 0.20
            root_causes["iOS cookie/session handling differences"] = 0.15
            root_causes["Safari security policies"] = 0.15
    
    # CORS issues
    # Note: CORS issues do not prevent token storage, only token transmission
    if "cors" in combined_text:
        root_causes["CORS credentials not configured (affects transmission, not storage)"] = 0.45
        root_causes["Origin mismatch in CORS policy"] = 0.30
        root_causes["Preflight request failing"] = 0.15
        root_causes["Server CORS headers incorrect"] = 0.10
    
    # Network/Request issues
    if "network" in combined_text or ("request" in combined_text and "fail" in combined_text):
        root_causes["Network connectivity issue"] = 0.35
        root_causes["API endpoint unreachable"] = 0.25
        root_causes["Request timeout"] = 0.20
        root_causes["Server error response"] = 0.20
    
    # Delete-specific issues
    if "delete" in combined_text and not root_causes:
        root_causes["Missing authentication for DELETE request"] = 0.40
        root_causes["DELETE method not allowed by CORS"] = 0.25
        root_causes["Incorrect API endpoint path"] = 0.20
        root_causes["Permission/authorization issue"] = 0.15
    
    # Sort by confidence score
    sorted_causes = dict(sorted(root_causes.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_causes


def format_confidence_scores(root_causes: dict) -> str:
    """
    Format confidence scores into readable output.
    
    Args:
        root_causes (dict): Root causes with confidence scores
    
    Returns:
        str: Formatted confidence analysis
    """
    if not root_causes:
        return ""
    
    output = "[ANALYSIS] Root Cause Likelihood\n\n"
    
    for cause, confidence in root_causes.items():
        percentage = int(confidence * 100)
        
        # Visual bar
        bar_length = int(confidence * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Confidence label
        if confidence >= 0.6:
            label = "VERY HIGH"
        elif confidence >= 0.5:
            label = "HIGH"
        elif confidence >= 0.25:
            label = "MEDIUM"
        else:
            label = "LOW"
        
        output += f"[{label:^9}] {bar} {percentage}%\n"
        output += f"          {cause}\n\n"
    
    output += "\nRecommendation: Start investigation with the highest confidence items.\n"
    
    return output


def generate_root_cause_analysis(user_input: str, conversation_history: str) -> str:
    """
    Generate complete root cause analysis with confidence scores.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        str: Formatted root cause analysis
    """
    root_causes = calculate_root_cause_confidence(user_input, conversation_history)
    
    if root_causes:
        return format_confidence_scores(root_causes)
    
    return ""
