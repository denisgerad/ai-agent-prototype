"""
Confirmation Gate Generator
Requires user confirmation before suggesting fixes to prevent false solutions
"""


def generate_confirmation_gate(issue_type: str, platform: str = None) -> dict:
    """
    Generate confirmation questions before suggesting fixes.
    
    Args:
        issue_type (str): Type of issue detected
        platform (str): Platform where issue occurs
    
    Returns:
        dict: Confirmation options and explanation
    """
    gate = {
        "required": True,
        "message": "",
        "options": [],
        "explanation": ""
    }
    
    # Token/Authentication issues
    if "token" in issue_type.lower() or "auth" in issue_type.lower():
        gate["message"] = "[CONFIRM] Confirmation Required Before Fix"
        gate["explanation"] = "To provide the most accurate solution, please confirm which scenario matches your situation:"
        gate["options"] = [
            {
                "id": "A",
                "description": "localStorage.getItem('token') returns null/undefined BEFORE delete attempt",
                "meaning": "Token is not being stored or retrieved properly"
            },
            {
                "id": "B",
                "description": "Token exists in storage BUT DELETE request has no Authorization header",
                "meaning": "Token retrieval works but isn't being attached to requests"
            },
            {
                "id": "C",
                "description": "Authorization header is present BUT server returns 401/403",
                "meaning": "Token is sent but server rejects it (expired/invalid/permissions)"
            },
            {
                "id": "D",
                "description": "Not sure yet - need to run verification tests first",
                "meaning": "Run the 60-90 second verification tests to determine"
            }
        ]
    
    # CORS issues
    elif "cors" in issue_type.lower():
        gate["message"] = "[CONFIRM] CORS Issue Confirmation"
        gate["explanation"] = "CORS errors can have different root causes. Please confirm:"
        gate["options"] = [
            {
                "id": "A",
                "description": "Console shows 'Access-Control-Allow-Origin' error",
                "meaning": "Server CORS configuration issue"
            },
            {
                "id": "B",
                "description": "Console shows 'credentials mode' error",
                "meaning": "Need to add credentials: 'include' to requests"
            },
            {
                "id": "C",
                "description": "Preflight OPTIONS request fails",
                "meaning": "Server doesn't handle OPTIONS requests properly"
            },
            {
                "id": "D",
                "description": "Not sure - no clear CORS error in console",
                "meaning": "May not actually be a CORS issue"
            }
        ]
    
    # Delete-specific issues
    elif "delete" in issue_type.lower():
        gate["message"] = "[CONFIRM] Delete Operation Diagnosis"
        gate["explanation"] = "Before suggesting a fix, please confirm the exact failure point:"
        gate["options"] = [
            {
                "id": "A",
                "description": "DELETE request never gets sent (nothing in Network tab)",
                "meaning": "Client-side JavaScript issue preventing request"
            },
            {
                "id": "B",
                "description": "DELETE request is sent but returns error (4xx/5xx)",
                "meaning": "Server receives request but rejects it"
            },
            {
                "id": "C",
                "description": "DELETE returns 200 OK but item doesn't actually delete",
                "meaning": "Server-side logic or database issue"
            },
            {
                "id": "D",
                "description": "Not sure - haven't checked Network tab yet",
                "meaning": "Need to inspect browser DevTools first"
            }
        ]
    
    # Network/Request issues
    elif "network" in issue_type.lower():
        gate["message"] = "[CONFIRM] Network Request Diagnosis"
        gate["explanation"] = "To narrow down the network issue:"
        gate["options"] = [
            {
                "id": "A",
                "description": "Request times out (no response)",
                "meaning": "Network connectivity or server unreachable"
            },
            {
                "id": "B",
                "description": "Request fails immediately with error",
                "meaning": "Could be CORS, SSL, or DNS issue"
            },
            {
                "id": "C",
                "description": "Request succeeds on retry",
                "meaning": "Intermittent connectivity or rate limiting"
            },
            {
                "id": "D",
                "description": "Need to check Network tab first",
                "meaning": "Run verification tests to gather more info"
            }
        ]
    
    return gate


def format_confirmation_gate(gate: dict) -> str:
    """
    Format confirmation gate into readable output.
    
    Args:
        gate (dict): Confirmation gate data
    
    Returns:
        str: Formatted confirmation request
    """
    if not gate["options"]:
        return ""
    
    output = gate['message'] + "\n"
    output += "="*70 + "\n\n"
    output += gate['explanation'] + "\n\n"
    
    for option in gate["options"]:
        output += f"**[{option['id']}]** {option['description']}\n"
        output += f"    -> {option['meaning']}\n\n"
    
    output += "="*70 + "\n"
    output += "\nPlease reply with: A, B, C, or D\n"
    output += "\nThis confirmation helps provide the most accurate fix for your specific situation.\n"
    
    return output


def should_require_confirmation(user_input: str, conversation_history: str) -> bool:
    """
    Determine if confirmation gate should be shown.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        bool: True if confirmation is needed
    """
    combined_text = (user_input + " " + conversation_history).lower()
    
    # Check if user has already provided detailed confirmation
    confirmation_indicators = [
        "console shows",
        "network tab shows",
        "returns null",
        "returns undefined",
        "header is missing",
        "header is present",
        "i checked",
        "i verified",
        "i confirmed"
    ]
    
    has_confirmation = any(indicator in combined_text for indicator in confirmation_indicators)
    
    # Check if this is a follow-up where user is providing more details
    is_follow_up = "user:" in conversation_history.lower() and "assistant:" in conversation_history.lower()
    
    # Require confirmation if:
    # 1. It's a follow-up with error details
    # 2. User hasn't already provided detailed confirmation
    if is_follow_up and not has_confirmation:
        # Check for error keywords that would trigger confirmation
        error_keywords = ["token", "cors", "delete", "network", "error", "fail"]
        has_error = any(keyword in combined_text for keyword in error_keywords)
        return has_error
    
    return False


def generate_confirmation_for_issue(user_input: str, conversation_history: str) -> str:
    """
    Generate confirmation gate based on conversation context.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        str: Formatted confirmation gate or empty string
    """
    if not should_require_confirmation(user_input, conversation_history):
        return ""
    
    combined_text = (user_input + " " + conversation_history).lower()
    
    # Determine issue type
    issue_type = ""
    if "token" in combined_text or "auth" in combined_text:
        issue_type = "token"
    elif "cors" in combined_text:
        issue_type = "cors"
    elif "delete" in combined_text:
        issue_type = "delete"
    elif "network" in combined_text:
        issue_type = "network"
    else:
        return ""
    
    # Determine platform
    platform = None
    if "ios" in combined_text or "safari" in combined_text:
        platform = "ios_safari"
    
    gate = generate_confirmation_gate(issue_type, platform)
    
    if gate["options"]:
        return format_confirmation_gate(gate)
    
    return ""
