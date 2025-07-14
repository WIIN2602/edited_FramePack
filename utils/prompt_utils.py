import os
import json
from langchain_core.prompts import PromptTemplate
def get_prompt_by_category(category: str):
    """
    Loads a prompt template file based on the given category.

    Parameters:
    category (str): The category used to select the appropriate prompt file
                    (e.g. "normal", "news"). If not found, defaults to "normal".

    Returns:
    PromptTemplate: The loaded prompt template based on the category
    """
    config_path = os.path.join("prompts", "prompt_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    file_name = mapping.get(category.lower(), mapping.get("normal"))
    prompt_path = os.path.join("prompts", file_name)
    return PromptTemplate.from_file(prompt_path)
