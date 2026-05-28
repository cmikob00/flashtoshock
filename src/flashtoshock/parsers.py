'''
Parser File for Flash to Shock
'''

# --- Helper Functions ---
def parse_input(val_str):
    """Safely converts strings to floats, parsing simple fractions if present."""
    try:
        if '/' in val_str:
            num, den = val_str.split('/')
            return float(num) / float(den)
        return float(val_str)
    except Exception:
        raise ValueError("Invalid numerical input")