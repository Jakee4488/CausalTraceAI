"""LLM-facing schemas and pipeline status.

Two of these are what Gemini fills through constrained decoding
(``ParsedIntervention``, ``ParsedCausalQuery``, ported from ``causal_model.ipynb``
cells 22 and 24); ``CausalStatus`` is the phase badge the UI renders.

The engine's *results* are deliberately not modelled here.
:class:`~src.causal.decision.CausalDecisionAPI` returns plain dicts whose every
value is already a Python ``float``/``str``/``list`` — JSON-safe as-is — and the
notebook-parity tests pin their exact shape. Wrapping them in pydantic would add
a second definition of that shape with nothing to validate (we produce them
ourselves) and one more place for the two to drift apart.

No torch, no langgraph, no vertexai imports: this module stays cheap and
hermetic so schema tests need none of them.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ParsedIntervention(BaseModel):
    """Intervention extracted from natural-language user input."""

    variable: str = Field(
        description=(
            "Name of the non-decision causal variable being intervened upon."
        )
    )

    value: float = Field(description="Value imposed by do(variable=value).")

    baseline_value: Optional[float] = Field(
        default=None,
        description=(
            "Optional comparison intervention value. For example, compare "
            "do(C=1) with do(C=0). Leave null when comparing against the "
            "ordinary stochastic model."
        ),
    )

    mode: Literal["fixed_policy", "reoptimise_policy"] = Field(
        description=(
            "Use fixed_policy when the user wants to hold one decision fixed. "
            "Use reoptimise_policy when the optimal decision should be "
            "recalculated."
        )
    )

    fixed_decision: Optional[int] = Field(
        default=None,
        description=(
            "Decision to hold fixed. Required only when mode is fixed_policy."
        ),
    )

    @model_validator(mode="after")
    def validate_intervention(self) -> "ParsedIntervention":
        if self.mode == "fixed_policy" and self.fixed_decision is None:
            raise ValueError("fixed_policy requires fixed_decision.")
        return self


class ParsedCausalQuery(BaseModel):
    """Structured interpretation of a natural-language query."""

    query_type: Literal[
        "optimal_policy",
        "intervention_effect",
        "posterior_summary",
        "clarification_required",
    ] = Field(description="The operation requested by the user.")

    context: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Cause values observed before choosing D. Only include values "
            "explicitly stated or clearly given by the user. Unknown variables "
            "must not be included."
        ),
    )

    intervention: Optional[ParsedIntervention] = Field(
        default=None,
        description="Intervention specification for intervention queries.",
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask when essential information is missing.",
    )

    @model_validator(mode="after")
    def validate_query(self) -> "ParsedCausalQuery":
        if self.query_type == "intervention_effect" and self.intervention is None:
            raise ValueError("intervention_effect requires an intervention.")
        if self.query_type == "clarification_required" and not self.clarification_question:
            raise ValueError("clarification_required needs a question.")
        return self


class CausalStatus(BaseModel):
    """Pipeline phase, stored in state and shown in the UI badge."""

    phase: Literal[
        "interpreting",
        "validating",
        "computing",
        "explaining",
        "complete",
        "clarification",
        "failed",
    ] = "interpreting"
    query_type: str = ""
    note: str = ""
