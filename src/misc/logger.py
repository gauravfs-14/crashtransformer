import logging
import os
import datetime
from typing import Optional


class CrashTransformerLogger:
    """
    Simple logger for CrashTransformer pipeline that saves logs to a logs folder.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save log files (default: "logs")
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate timestamped log filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"crashtransformer_{timestamp}.log")
        
        # Setup logger
        self.logger = logging.getLogger("crashtransformer")
        self.logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_crash_processing(self, crash_id: str, stage: str, details: Optional[str] = None):
        """Log crash processing stages."""
        message = f"Processing Crash {crash_id} - Stage: {stage}"
        if details:
            message += f" - {details}"
        self.info(message)
    
    def log_llm_call(self, crash_id: str, provider: str, model: str, tokens: int, cost: float):
        """Log LLM API calls."""
        self.info(f"LLM Call - Crash {crash_id} | {provider}/{model} | Tokens: {tokens} | Cost: ${cost:.4f}")
    
    def log_summary_generation(self, crash_id: str, model: str, candidates: int, best_score: float):
        """Log summary generation."""
        self.info(f"Summary Generated - Crash {crash_id} | Model: {model} | Candidates: {candidates} | Best Score: {best_score:.3f}")
    
    def log_batch_completion(self, total_crashes: int, successful: int, failed: int, total_cost: float):
        """Log batch processing completion."""
        self.info(f"Batch Complete - Total: {total_crashes} | Success: {successful} | Failed: {failed} | Total Cost: ${total_cost:.2f}")
    
    def get_log_file_path(self) -> str:
        """Get the current log file path."""
        return self.log_file


# Convenience function to get a logger instance
def get_logger(log_dir: str = "logs", log_level: str = "INFO") -> CrashTransformerLogger:
    """
    Get a logger instance.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        
    Returns:
        CrashTransformerLogger instance
    """
    return CrashTransformerLogger(log_dir=log_dir, log_level=log_level)


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_logger(log_dir="logs", log_level="INFO")
    
    # Test logging
    logger.info("CrashTransformer Logger initialized")
    logger.log_crash_processing("19955047", "graph_extraction", "Extracting entities and relationships")
    logger.log_llm_call("19955047", "openai", "gpt-4o-mini", 150, 0.0023)
    logger.log_summary_generation("19955047", "facebook/bart-base", 4, 0.89)
    logger.log_batch_completion(100, 98, 2, 15.67)
    
    print(f"Log file created at: {logger.get_log_file_path()}")
