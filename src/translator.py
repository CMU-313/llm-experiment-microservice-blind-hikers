from src.llm import query_llm_robust

def translate_content(content: str) -> tuple[bool, str]:
    return query_llm_robust(content)
