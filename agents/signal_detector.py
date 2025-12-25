"""
Signal Detector Module
Detects stability and environment signals in user bug reports
to prevent unnecessary code refactoring.
"""


# Category A: Stability Signals
CATEGORY_A_SIGNALS = [
    "works well",
    "works fine",
    "mostly works",
    "sometimes",
    "intermittent",
    "only when",
    "works on",
    "works in",
    "doesn't work on",
    "does not work on"
]

# Category B: Environment Signals
CATEGORY_B_SIGNALS = [
    "on pc",
    "on mobile",
    "on my pc",
    "on my mobile",
    "on desktop",
    "on phone",
    "browser",
    "android",
    "ios",
    "iphone",
    "safari",
    "chrome",
    "firefox",
    "tab out",
    "tab in",
    "focus",
    "blur",
    "device",
    "platform"
]


def detect_signals(user_input: str) -> dict:
    """
    Detect stability and environment signals in user input.
    
    Args:
        user_input (str): The user's bug report or issue description
        
    Returns:
        dict: Dictionary with detected signals:
            - category_a (bool): True if stability signals detected
            - category_b (bool): True if environment signals detected
            - detected_keywords (list): List of matched keywords
    """
    user_input_lower = user_input.lower()
    
    detected_a = []
    detected_b = []
    
    # Check for Category A signals
    for signal in CATEGORY_A_SIGNALS:
        if signal in user_input_lower:
            detected_a.append(signal)
    
    # Check for Category B signals
    for signal in CATEGORY_B_SIGNALS:
        if signal in user_input_lower:
            detected_b.append(signal)
    
    return {
        "category_a": len(detected_a) > 0,
        "category_b": len(detected_b) > 0,
        "detected_keywords": detected_a + detected_b,
        "category_a_keywords": detected_a,
        "category_b_keywords": detected_b
    }


def should_investigate(signals: dict) -> bool:
    """
    Determine if investigation mode should be triggered.
    
    Args:
        signals (dict): Output from detect_signals()
        
    Returns:
        bool: True if investigation questions should be asked
    """
    return signals["category_a"] or signals["category_b"]
