"""
Verification Command Generator
Generates immediate verification commands for quick testing
"""


def generate_immediate_tests(platform: str, issue_type: str) -> dict:
    """
    Generate immediate verification tests based on platform and issue.
    
    Args:
        platform (str): Platform (e.g., "ios_safari", "android_chrome", "desktop")
        issue_type (str): Type of issue (e.g., "token", "cors", "network")
    
    Returns:
        dict: Test steps with timing and commands
    """
    tests = {
        "duration": "60 seconds",
        "steps": []
    }
    
    # iOS Safari specific tests
    if "ios" in platform.lower() or "safari" in platform.lower():
        if "token" in issue_type.lower() or "auth" in issue_type.lower():
            tests["steps"] = [
                {
                    "step": 1,
                    "action": "Open Safari on iPhone",
                    "detail": "Navigate to your app"
                },
                {
                    "step": 2,
                    "action": "Open DevTools (if connected to Mac)",
                    "detail": "Safari → Develop → [Your iPhone] → [Your App]"
                },
                {
                    "step": 3,
                    "action": "Console → Run:",
                    "command": "localStorage.getItem('token') || localStorage.getItem('authToken')",
                    "expected": "Should return token string"
                },
                {
                    "step": 4,
                    "action": "Trigger delete action",
                    "detail": "Try to delete a conversation"
                },
                {
                    "step": 5,
                    "action": "Console → Re-run:",
                    "command": "localStorage.getItem('token')",
                    "expected": "Check if token still exists"
                },
                {
                    "step": 6,
                    "action": "Network tab → Find DELETE request",
                    "detail": "Look for DELETE /api/conversations/..."
                },
                {
                    "step": 7,
                    "action": "Check Request Headers",
                    "detail": "Look for 'Authorization: Bearer [token]' header",
                    "expected": "Header should be present with valid token"
                }
            ]
            tests["duration"] = "90 seconds"
    
    # Android Chrome tests
    elif "android" in platform.lower():
        if "token" in issue_type.lower():
            tests["steps"] = [
                {
                    "step": 1,
                    "action": "Open Chrome on Android",
                    "detail": "Navigate to your app"
                },
                {
                    "step": 2,
                    "action": "Enable USB debugging",
                    "detail": "chrome://inspect on desktop"
                },
                {
                    "step": 3,
                    "action": "Console → Run:",
                    "command": "localStorage.getItem('token')",
                    "expected": "Should return token"
                },
                {
                    "step": 4,
                    "action": "Trigger delete action",
                    "detail": "Verify it works"
                },
                {
                    "step": 5,
                    "action": "Compare with iOS behavior",
                    "detail": "Document any differences"
                }
            ]
    
    # CORS tests
    if "cors" in issue_type.lower():
        tests["steps"].extend([
            {
                "step": len(tests["steps"]) + 1,
                "action": "Check Console for CORS errors",
                "detail": "Look for 'Access-Control-Allow-Origin' errors"
            },
            {
                "step": len(tests["steps"]) + 2,
                "action": "Network tab → Response Headers",
                "command": "Check for: Access-Control-Allow-Credentials: true",
                "expected": "Should be present"
            }
        ])
    
    return tests


def format_verification_tests(tests: dict, platform: str) -> str:
    """
    Format verification tests into readable output.
    
    Args:
        tests (dict): Test steps and duration
        platform (str): Platform name
    
    Returns:
        str: Formatted test instructions
    """
    output = f"[VERIFY] {tests['duration']} Verification Test ({platform})\n\n"
    
    for step in tests["steps"]:
        output += f"{step['step']}. **{step['action']}**\n"
        if 'detail' in step:
            output += f"   {step['detail']}\n"
        if 'command' in step:
            output += f"   Command: `{step['command']}`\n"
        if 'expected' in step:
            output += f"   Expected: {step['expected']}\n"
        output += "\n"
    
    return output


def generate_verification_for_issue(user_input: str, conversation_history: str) -> str:
    """
    Generate verification tests based on conversation context.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        str: Formatted verification tests or empty string
    """
    combined_text = (user_input + " " + conversation_history).lower()
    
    # Determine platform
    platform = "Desktop Browser"
    if "ios" in combined_text or "iphone" in combined_text:
        platform = "iOS Safari"
    elif "android" in combined_text:
        platform = "Android Chrome"
    
    # Determine issue type
    issue_type = ""
    if "token" in combined_text or "auth" in combined_text:
        issue_type = "token"
    elif "cors" in combined_text:
        issue_type = "cors"
    elif "network" in combined_text:
        issue_type = "network"
    else:
        return ""
    
    tests = generate_immediate_tests(platform, issue_type)
    
    if tests["steps"]:
        return format_verification_tests(tests, platform)
    
    return ""
