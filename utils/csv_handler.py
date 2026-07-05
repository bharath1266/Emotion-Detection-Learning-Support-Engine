from typing import Optional

from utils.history_logger import log_entry


def save_to_csv(user_input: str, emotion: Optional[str], response: str) -> None:
    """Compatibility wrapper that logs a simple entry to history.

    This function delegates to `history_logger.log_entry` to keep a single
    source of truth for CSV history management.
    """
    # Use emotional label as both bilstm and bert are unknown in old flow
    log_entry(user_input, None, None, emotion, None)