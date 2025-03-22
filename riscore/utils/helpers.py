"""Utility functions for RISCORE framework."""

from typing import List, Dict, Any, Optional
import re
from collections import Counter
import nltk
try:
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
except:
    pass


def count_tokens(text: str) -> int:
    """
    Count tokens in text using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    try:
        tokens = word_tokenize(text)
        return len(tokens)
    except:
        # Fallback to simple whitespace splitting
        return len(text.split())


def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract answer letter (A, B, C, D) from text.
    
    Args:
        text: Text containing answer
        
    Returns:
        Extracted letter or None
    """
    patterns = [
        r"(?:answer|Answer|ANSWER)\s*(?:is|:)?\s*([A-D])",
        r"(?:correct answer|option)\s*(?:is|:)?\s*([A-D])",
        r"\b([A-D])\s*(?:is correct|is the answer)",
        r"^([A-D])\s*$",  # Single letter answer
        r"\*\*([A-D])\*\*",  # Bold letter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Try to find first occurrence of A, B, C, or D
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()
    
    return None


def format_choices(choices: List[str]) -> str:
    """
    Format multiple choice options.
    
    Args:
        choices: List of answer choices
        
    Returns:
        Formatted string
    """
    return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])


def parse_model_response(
    response: str,
    extract_cot: bool = False
) -> Dict[str, str]:
    """
    Parse model response to extract answer and reasoning.
    
    Args:
        response: Raw model response
        extract_cot: Whether to extract chain-of-thought reasoning
        
    Returns:
        Dictionary with 'answer' and optionally 'reasoning'
    """
    result = {}
    
    # Extract answer
    answer = extract_answer_letter(response)
    result["answer"] = answer or ""
    
    # Extract reasoning if requested
    if extract_cot:
        # Try to find reasoning before the answer
        reasoning_patterns = [
            r"(?:Reasoning|Explanation|Let's think step by step):\s*(.+?)(?:Answer:|$)",
            r"(.+?)(?:Therefore|Thus|So),?\s*(?:the answer is|answer:)",
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result["reasoning"] = match.group(1).strip()
                break
        
        if "reasoning" not in result:
            # If no pattern matches, use the whole response minus the answer
            result["reasoning"] = response.strip()
    
    return result


def majority_vote(answers: List[str]) -> str:
    """
    Get majority vote from list of answers.
    
    Args:
        answers: List of answer letters
        
    Returns:
        Most common answer
    """
    if not answers:
        return ""
    
    # Filter out empty answers
    valid_answers = [a for a in answers if a]
    
    if not valid_answers:
        return ""
    
    counter = Counter(valid_answers)
    return counter.most_common(1)[0][0]


def calculate_agreement(answers: List[str]) -> float:
    """
    Calculate agreement score (consistency) among answers.
    
    Args:
        answers: List of answer letters
        
    Returns:
        Agreement score (0-1)
    """
    if not answers:
        return 0.0
    
    valid_answers = [a for a in answers if a]
    
    if not valid_answers:
        return 0.0
    
    counter = Counter(valid_answers)
    most_common_count = counter.most_common(1)[0][1]
    
    return most_common_count / len(valid_answers)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Chunk list into smaller lists.
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def keep_single_ids(id_list: List[str]) -> List[str]:
    """
    Keep only unique IDs, removing duplicates.
    
    Args:
        id_list: List of IDs (may contain duplicates)
        
    Returns:
        List with unique IDs only
    """
    seen = set()
    result = []
    
    for id in id_list:
        if id not in seen:
            seen.add(id)
            result.append(id)
    
    return result


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list structure
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


class TextFormatter:
    """Utility class for text formatting."""
    
    @staticmethod
    def wrap_text(text: str, width: int = 80) -> str:
        """Wrap text to specified width."""
        import textwrap
        return textwrap.fill(text, width=width)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean excessive whitespace from text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def remove_special_tokens(text: str) -> str:
        """Remove special tokens from model output."""
        # Remove common special tokens
        special_tokens = [
            '<s>', '</s>',
            '<|begin_of_text|>', '<|end_of_text|>',
            '<|start_header_id|>', '<|end_header_id|>',
            '<|eot_id|>',
            '[INST]', '[/INST]',
            '<start_of_turn>', '<end_of_turn>',
            '<|system|>', '<|user|>', '<|assistant|>',
            '<|end|>',
        ]
        
        for token in special_tokens:
            text = text.replace(token, '')
        
        return text.strip()


class Logger:
    """Simple logger utility."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        """
        Initialize logger.
        
        Args:
            log_file: Path to log file (None for console only)
            verbose: Whether to print to console
        """
        self.log_file = log_file
        self.verbose = verbose
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        if self.verbose:
            print(formatted_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message + '\n')
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")
