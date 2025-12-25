"""
Code Inspection Generator
Generates specific code locations and checks based on the bug analysis
"""


def generate_code_inspection_steps(issue_type: str, platform: str = None, tech_stack: str = None) -> list:
    """
    Generate specific code inspection steps based on the issue.
    
    Args:
        issue_type (str): Type of issue (e.g., "token_error", "network_failure", "cors")
        platform (str): Platform where issue occurs (e.g., "ios_safari", "android_chrome")
        tech_stack (str): Technology stack (e.g., "mern", "react", "vue")
    
    Returns:
        list: Specific code locations and checks to perform
    """
    steps = []
    
    # Token/Authentication related issues
    if "token" in issue_type.lower() or "auth" in issue_type.lower():
        steps.extend([
            {
                "file": "Client-side authentication handling (axios.js, api.js, or similar)",
                "check": "Search for localStorage.getItem('token') or cookies.get('token')",
                "command": "console.log('Token:', localStorage.getItem('token'))"
            },
            {
                "file": "API interceptor configuration",
                "check": "Verify axios/fetch interceptor attaches Authorization header",
                "command": "Check request.headers['Authorization'] or headers.authorization"
            },
            {
                "file": "Delete API call implementation",
                "check": "Locate the DELETE call for conversations/messages",
                "command": "Verify Authorization header is attached to this specific request"
            }
        ])
        
        if platform and "ios" in platform.lower():
            steps.append({
                "file": "Token storage implementation",
                "check": "Check if localStorage is available on iOS Safari (may be blocked in private mode)",
                "command": "Test: typeof localStorage !== 'undefined' && localStorage.getItem('token')"
            })
    
    # CORS related issues
    if "cors" in issue_type.lower():
        steps.extend([
            {
                "file": "Server CORS configuration (server.js, app.js)",
                "check": "Verify CORS credentials setting: credentials: true",
                "command": "cors({ origin: true, credentials: true })"
            },
            {
                "file": "Client API calls",
                "check": "Verify fetch/axios uses credentials: 'include'",
                "command": "fetch(url, { credentials: 'include' })"
            }
        ])
    
    # Network/Request issues
    if "network" in issue_type.lower() or "request" in issue_type.lower():
        steps.extend([
            {
                "file": "API endpoint implementation",
                "check": "Verify DELETE endpoint exists and is accessible",
                "command": "Check server routes for DELETE /api/conversations/:id"
            },
            {
                "file": "Request headers",
                "check": "Inspect all headers being sent with the request",
                "command": "Console → Network tab → DELETE request → Headers"
            }
        ])
    
    return steps


def format_inspection_checklist(steps: list) -> str:
    """
    Format inspection steps into a readable checklist.
    
    Args:
        steps (list): List of inspection steps
    
    Returns:
        str: Formatted checklist
    """
    output = "[INSPECTION] Code Inspection Checklist\n\n"
    
    for i, step in enumerate(steps, 1):
        output += f"{i}. **{step['file']}**\n"
        output += f"   - Check: {step['check']}\n"
        if 'command' in step:
            output += f"   - Command/Test: `{step['command']}`\n"
        output += "\n"
    
    return output


def generate_inspection_for_conversation(user_input: str, conversation_history: str) -> str:
    """
    Analyze conversation and generate appropriate inspection steps.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        str: Formatted inspection checklist or empty string if not applicable
    """
    combined_text = (user_input + " " + conversation_history).lower()
    
    # Determine issue type
    issue_type = ""
    if "token" in combined_text or "auth" in combined_text:
        issue_type = "token_error"
    elif "cors" in combined_text:
        issue_type = "cors"
    elif "network" in combined_text or "request" in combined_text:
        issue_type = "network_failure"
    else:
        return ""
    
    # Determine platform
    platform = None
    if "ios" in combined_text or "iphone" in combined_text or "safari" in combined_text:
        platform = "ios_safari"
    elif "android" in combined_text:
        platform = "android_chrome"
    
    # Determine tech stack
    tech_stack = None
    if "mern" in combined_text or "react" in combined_text:
        tech_stack = "mern"
    
    steps = generate_code_inspection_steps(issue_type, platform, tech_stack)
    
    if steps:
        return format_inspection_checklist(steps)
    
    return ""
