from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResearchContext:
    agent_id: str
    world_spec: Dict[str, Any] = field(default_factory=dict)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    local_notes: List[Dict[str, Any]] = field(default_factory=list)
    shared_notes: List[Dict[str, Any]] = field(default_factory=list)
    local_observations: List[Dict[str, Any]] = field(default_factory=list)
    shared_observations: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis: List[str] = field(default_factory=list)
    data_card: Optional[Dict[str, Any]] = None
    method_card: Optional[Dict[str, Any]] = None
    plan_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorInput:
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorOutput:
    ok: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=dict)
    error_code: str = ""
    precondition_failed: bool = False


@dataclass
class FailureDiagnosis:
    error_class: str = "unknown"
    error_codes: List[str] = field(default_factory=list)
    severity: str = "medium"
    retryable: bool = True
    root_cause: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    repair_hints: List[str] = field(default_factory=list)


@dataclass
class RecoveryDecision:
    should_create_tasks: bool = False
    recovery_tasks: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    rag_degraded_pause_active: bool = False


@dataclass
class EvidenceItem:
    evidence_id: str
    source_collection: str
    source_type: str
    source_id: str
    tags: List[str] = field(default_factory=list)
    chunk_text: str = ""
    match_score: float = 0.0
    retrieval_mode: str = "vector"


@dataclass
class MethodCardQuality:
    schema_valid: bool = False
    citation_coverage: float = 0.0
    executable_minimum: bool = False
    level: str = "L2"
    degraded: bool = True
    fail_reasons: List[str] = field(default_factory=list)
