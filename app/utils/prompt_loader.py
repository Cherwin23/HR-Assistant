# Load prompt from file
def load_prompt(path: str) -> str:
    """Load a text prompt from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:        
        raise FileNotFoundError(f"Prompt not found. Tried: {path}")