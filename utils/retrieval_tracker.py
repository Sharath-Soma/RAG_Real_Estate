"""Retrieval progress tracking for enhanced user experience."""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class RetrievalStage(str, Enum):
    """Stages of the retrieval process."""

    SEARCHING = "Searching knowledge base..."
    RETRIEVING = "Retrieving documents..."
    RANKING = "Ranking documents..."
    GENERATING = "Generating answer..."
    COMPLETE = "Complete"


@dataclass
class RetrievalProgress:
    """Tracks retrieval progress for UI display."""

    stage: RetrievalStage
    percentage: int
    message: str
    details: Optional[str] = None


class RetrievalTracker:
    """Tracks and visualizes retrieval progress."""

    def __init__(self):
        """Initialize the tracker."""
        self.current_stage = RetrievalStage.SEARCHING
        self.stages = [
            (RetrievalStage.SEARCHING, "Searching knowledge base...", 10),
            (RetrievalStage.RETRIEVING, "Retrieving documents...", 35),
            (RetrievalStage.RANKING, "Ranking documents...", 65),
            (RetrievalStage.GENERATING, "Generating answer...", 90),
            (RetrievalStage.COMPLETE, "Complete", 100),
        ]
        self.current_index = 0

    def next_stage(self, details: Optional[str] = None) -> RetrievalProgress:
        """Move to next retrieval stage."""
        if self.current_index < len(self.stages):
            stage, message, percentage = self.stages[self.current_index]
            self.current_stage = stage
            self.current_index += 1
            return RetrievalProgress(
                stage=stage,
                percentage=percentage,
                message=message,
                details=details,
            )
        return RetrievalProgress(
            stage=RetrievalStage.COMPLETE,
            percentage=100,
            message="Complete",
            details=details,
        )

    def get_current_progress(self, details: Optional[str] = None) -> RetrievalProgress:
        """Get current progress state."""
        if self.current_index > 0 and self.current_index <= len(self.stages):
            stage, message, percentage = self.stages[self.current_index - 1]
            return RetrievalProgress(
                stage=stage,
                percentage=percentage,
                message=message,
                details=details,
            )
        return RetrievalProgress(
            stage=RetrievalStage.SEARCHING,
            percentage=0,
            message="Starting...",
            details=details,
        )

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.current_index = 0
        self.current_stage = RetrievalStage.SEARCHING
