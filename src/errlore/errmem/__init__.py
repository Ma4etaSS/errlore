"""Error memory (Amygdala) — model weakness tracking, pattern detection, prompt injection.

Public API:
    classify_error    — regex extraction of ErrorType from exception/stacktrace/message
    ErrorTracker      — record_error / get_model_profile, persisted via errlore.io
    PatternDetector   — group (model, error_type, task_type), similarity with RU/EN stemming
    WarningInjector   — build_warning / get_penalty for prompt injection
    sanitize_description — strip raw JSON / overlong text from descriptions
"""

from errlore.errmem.classifier import classify_error
from errlore.errmem.injector import WarningInjector
from errlore.errmem.patterns import PatternDetector
from errlore.errmem.tracker import ErrorTracker

__all__ = [
    "ErrorTracker",
    "PatternDetector",
    "WarningInjector",
    "classify_error",
]
