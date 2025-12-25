"""
Prompt Builder Module
Builds analysis prompts incorporating AI debugging instructions
and detected signals.
"""


DEBUG_INSTRUCTIONS = """
# AI Debugging Instructions

## Core Principles
You are acting as a senior engineer analyzing a bug report.

### CRITICAL: Error Message Priority
**ALWAYS prioritize specific error messages over general platform differences!**

If the user mentions ANY error message, console output, or specific behavior:
1. Extract and highlight the exact error/message
2. Focus your entire analysis on that specific error
3. Research what causes that exact error on that platform
4. Provide targeted solutions for that specific error

Do NOT give generic platform compatibility advice when there's a specific error message!

### Analysis Hierarchy (in order of priority)
1. **Specific error messages** ("no token", "undefined", "CORS error", etc.)
2. **Console logs or network request failures**
3. **Specific behavior differences** (button doesn't respond, data not saved)
4. **General platform differences** (only consider if no specific error)

### Prohibited Behavior
- Do NOT give generic "check browser compatibility" advice when there's a specific error
- Do NOT refactor working logic
- Do NOT rewrite components without evidence
- Do NOT suggest "best practices" fixes without relevance
- Do NOT ignore error messages in favor of generic platform advice

### Analysis Approach
1. **Extract specific errors first** - Look for error messages, console output, network failures
2. **Diagnose the specific error** - What causes this exact error?
3. **Narrow to root cause** - Focus on the error, not generic issues
4. **Suggest verification steps** - How to confirm this diagnosis
5. **Provide targeted fixes** - Fix the specific error, not general code

### Code Change Criteria
Only suggest code changes if:
- The issue is reproducible across environments
- The logic is incorrect in all cases
- External causes have been ruled out

---
"""


def build_prompt(user_input: str, code_context: str = None, signals: dict = None) -> str:
    """
    Build an analysis prompt with debugging instructions.
    
    Args:
        user_input (str): The user's bug report
        code_context (str): Optional code context to analyze
        signals (dict): Detected signals from signal_detector
        
    Returns:
        str: Complete prompt for the LLM
    """
    prompt = DEBUG_INSTRUCTIONS
    
    # Add signal context if available
    if signals and (signals.get("category_a") or signals.get("category_b")):
        prompt += "## Detected Signals\n\n"
        
        if signals.get("category_a"):
            prompt += "**Stability Signals Detected:**\n"
            prompt += "The user mentioned: " + ", ".join(f"'{k}'" for k in signals.get("category_a_keywords", [])) + "\n"
            prompt += "→ This suggests the core logic is likely correct. Focus on triggering conditions.\n\n"
        
        if signals.get("category_b"):
            prompt += "**Environment Signals Detected:**\n"
            prompt += "The user mentioned: " + ", ".join(f"'{k}'" for k in signals.get("category_b_keywords", [])) + "\n"
            prompt += "→ This suggests a platform or interaction issue, not a logic error.\n\n"
        
        prompt += "---\n\n"
    
    # Add the user's bug report with emphasis on error extraction
    prompt += "## User's Bug Report\n\n"
    prompt += user_input + "\n\n"
    
    # Check for error messages or specific technical details
    error_keywords = ['error', 'message', 'console', 'exception', 'failed', 'undefined', 'null', 'token', 'cors', 'network', 'status', 'code']
    has_error_info = any(keyword in user_input.lower() for keyword in error_keywords)
    
    if has_error_info:
        prompt += "**CRITICAL: The user has mentioned specific error details above. Extract and analyze these FIRST before considering generic solutions.**\n\n"
    
    # Add code context if provided
    if code_context:
        prompt += "## Code Context\n\n"
        prompt += "```\n" + code_context + "\n```\n\n"
    
    # Add analysis request
    prompt += "## Your Task\n\n"
    prompt += "Based on the signals detected and the information provided:\n"
    prompt += "1. Assess whether this is likely a logic error or an environment/interaction issue\n"
    prompt += "2. If more information is needed, ask specific clarifying questions\n"
    prompt += "3. If you can diagnose the issue, provide:\n"
    prompt += "   - Root cause analysis\n"
    prompt += "   - Verification steps the user can take\n"
    prompt += "   - Minimal, targeted fix (only if code change is truly needed)\n"
    prompt += "\nRemember: DO NOT refactor working code. Focus on the actual problem.\n"
    
    return prompt


def build_analysis_prompt_with_answers(user_input: str, code_context: str, 
                                      clarifications: dict) -> str:
    """
    Build a prompt after clarification questions have been answered.
    
    Args:
        user_input (str): Original bug report
        code_context (str): Code to analyze
        clarifications (dict): Answers to clarification questions
        
    Returns:
        str: Complete analysis prompt
    """
    prompt = DEBUG_INSTRUCTIONS
    
    prompt += "## Original Issue\n\n"
    prompt += user_input + "\n\n"
    
    prompt += "## Clarifications Provided\n\n"
    
    # Extract and highlight error messages from clarifications
    error_messages = []
    for question, answer in clarifications.items():
        prompt += f"**Q:** {question}\n"
        prompt += f"**A:** {answer}\n\n"
        # Check if answer contains error keywords
        error_keywords = ['error', 'message', 'console', 'exception', 'failed', 'undefined', 'null', 'token', 'cors']
        if any(keyword in answer.lower() for keyword in error_keywords):
            error_messages.append(answer)
    
    if error_messages:
        prompt += "\n**CRITICAL ERROR DETAILS FOUND:**\n"
        for err in error_messages:
            prompt += f"- {err}\n"
        prompt += "\n**Your analysis MUST focus on these specific errors, not generic platform differences!**\n\n"
    
    if code_context:
        prompt += "## Code Context\n\n"
        prompt += "```\n" + code_context + "\n```\n\n"
    
    prompt += "## Your Task\n\n"
    prompt += "Now that you have complete context:\n"
    prompt += "1. Provide a root cause analysis\n"
    prompt += "2. Suggest verification steps\n"
    prompt += "3. If a code change is needed, provide a minimal, targeted fix\n"
    prompt += "4. If this is an environment issue, suggest configuration or usage changes\n"
    
    return prompt
