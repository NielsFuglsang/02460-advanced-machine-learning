import json

def read_json(filename):
    """Reads json file and returns as dict.
    Args:
        filename (str): Filename.
    Returns:
        Dictionary containing json content.
    """
    with open(filename) as file_in:
        return json.load(file_in)