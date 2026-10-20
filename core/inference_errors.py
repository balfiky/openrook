"""Research resource-control signals, separate from optional inference failures."""


class ResearchBudgetExceeded(RuntimeError):
    def __init__(self, resume_at: float):
        self.resume_at = resume_at
        super().__init__(f"Inference resource period exhausted; resumes at {resume_at:.0f}")


class ResearchStudyLimit(RuntimeError):
    """A permanent, recorded resource stop for one bounded study."""
