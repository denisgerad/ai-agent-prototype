"""
Fix Strategy Generator
Generates multiple fix options without immediately refactoring code
"""


def generate_fix_strategies(issue_type: str, platform: str = None) -> list:
    """
    Generate multiple fix strategies for the issue.
    
    Args:
        issue_type (str): Type of issue (e.g., "token_storage", "cors", "network")
        platform (str): Affected platform
    
    Returns:
        list: Fix strategies with pros/cons
    """
    strategies = []
    
    # Token storage issues on iOS Safari
    if "token" in issue_type.lower() and platform and "ios" in platform.lower():
        strategies = [
            {
                "name": "Option A: Use httpOnly cookies (Recommended)",
                "description": "Store authentication token in httpOnly cookies instead of localStorage",
                "pros": [
                    "Works reliably on all platforms including iOS Safari",
                    "More secure (not accessible via JavaScript)",
                    "Immune to XSS attacks",
                    "No private browsing issues"
                ],
                "cons": [
                    "Requires server-side changes to set cookies",
                    "Need to configure CORS credentials properly",
                    "Slightly more complex CSRF protection needed"
                ],
                "implementation_hint": "Server: res.cookie('token', token, { httpOnly: true, secure: true })\nClient: Use credentials: 'include' in fetch/axios"
            },
            {
                "name": "Option B: sessionStorage fallback with detection",
                "description": "Detect localStorage availability and fall back to sessionStorage",
                "pros": [
                    "Client-side only change",
                    "Works in most scenarios",
                    "Quick to implement"
                ],
                "cons": [
                    "sessionStorage cleared when tab closes",
                    "Still fails in iOS private browsing",
                    "Poor UX (users need to re-login often)"
                ],
                "implementation_hint": "const storage = typeof localStorage !== 'undefined' ? localStorage : sessionStorage"
            },
            {
                "name": "Option C: In-memory storage with IndexedDB backup",
                "description": "Store token in memory with IndexedDB as persistent backup",
                "pros": [
                    "Works on all platforms",
                    "Good performance",
                    "Fallback mechanism"
                ],
                "cons": [
                    "More complex implementation",
                    "Need to handle IndexedDB API",
                    "Token lost on page refresh (if IndexedDB blocked)"
                ],
                "implementation_hint": "Use Dexie.js or localForage for easier IndexedDB access"
            }
        ]
    
    # CORS issues
    elif "cors" in issue_type.lower():
        strategies = [
            {
                "name": "Option A: Configure server CORS with credentials",
                "description": "Properly configure CORS to allow credentials",
                "pros": [
                    "Proper solution for authentication",
                    "Works across all browsers",
                    "Industry standard"
                ],
                "cons": [
                    "Requires server configuration",
                    "Need to specify exact origins (can't use wildcard with credentials)"
                ],
                "implementation_hint": "cors({ origin: 'https://yourapp.com', credentials: true })"
            },
            {
                "name": "Option B: Proxy API requests through same origin",
                "description": "Use a server-side proxy to avoid CORS",
                "pros": [
                    "No CORS issues",
                    "Can add authentication at proxy level",
                    "Works with any backend"
                ],
                "cons": [
                    "Additional infrastructure",
                    "Extra latency",
                    "More complex deployment"
                ],
                "implementation_hint": "Use Next.js API routes or Express proxy middleware"
            }
        ]
    
    # Delete-specific issues
    elif "delete" in issue_type.lower():
        strategies = [
            {
                "name": "Option A: Verify Authorization header attachment",
                "description": "Ensure DELETE requests include proper authentication",
                "pros": [
                    "Direct fix to the root cause",
                    "Minimal code changes",
                    "Works across platforms"
                ],
                "cons": [
                    "Need to identify where header is missing",
                    "May require interceptor update"
                ],
                "implementation_hint": "axios.interceptors.request.use(config => { config.headers.Authorization = `Bearer ${token}`; return config; })"
            },
            {
                "name": "Option B: Use axios/fetch interceptor globally",
                "description": "Set up global interceptor to add auth to all requests",
                "pros": [
                    "Applies to all requests automatically",
                    "Centralized authentication logic",
                    "Easy to maintain"
                ],
                "cons": [
                    "May add auth to requests that don't need it",
                    "Need to handle token refresh"
                ],
                "implementation_hint": "Configure once in your API client setup file"
            }
        ]
    
    return strategies


def format_fix_strategies(strategies: list) -> str:
    """
    Format fix strategies into readable output.
    
    Args:
        strategies (list): List of fix strategies
    
    Returns:
        str: Formatted strategies
    """
    if not strategies:
        return ""
    
    output = "[SOLUTIONS] Fix Strategy Options\n\n"
    output += "Choose the approach that best fits your requirements:\n\n"
    
    for i, strategy in enumerate(strategies, 1):
        output += f"{'='*60}\n"
        output += f"{strategy['name']}\n"
        output += f"{'='*60}\n\n"
        
        output += f"**Description:** {strategy['description']}\n\n"
        
        output += "**Pros:**\n"
        for pro in strategy['pros']:
            output += f"  + {pro}\n"
        output += "\n"
        
        output += "**Cons:**\n"
        for con in strategy['cons']:
            output += f"  - {con}\n"
        output += "\n"
        
        if 'implementation_hint' in strategy:
            output += f"**Implementation Hint:**\n```\n{strategy['implementation_hint']}\n```\n\n"
    
    return output


def generate_fix_strategies_for_issue(user_input: str, conversation_history: str) -> str:
    """
    Generate fix strategies based on conversation context.
    
    Args:
        user_input (str): Current user input
        conversation_history (str): Full conversation history
    
    Returns:
        str: Formatted fix strategies
    """
    combined_text = (user_input + " " + conversation_history).lower()
    
    # Determine issue type
    issue_type = ""
    if "token" in combined_text:
        issue_type = "token_storage"
    elif "cors" in combined_text:
        issue_type = "cors"
    elif "delete" in combined_text:
        issue_type = "delete_request"
    else:
        return ""
    
    # Determine platform
    platform = None
    if "ios" in combined_text or "safari" in combined_text:
        platform = "ios_safari"
    
    strategies = generate_fix_strategies(issue_type, platform)
    
    if strategies:
        return format_fix_strategies(strategies)
    
    return ""
