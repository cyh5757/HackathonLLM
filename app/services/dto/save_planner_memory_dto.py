from dataclasses import dataclass


@dataclass
class SavePlannerWorkerPlanDto:
    number: int
    worker_name: str
    request: str


@dataclass
class SavePlannerWorkerDto:
    order: int
    worker_name: str
    response: str | None = None


@dataclass
class SavePlannerMemoryDto:
    worker_plan: list[SavePlannerWorkerPlanDto]
    done_workers: list[SavePlannerWorkerDto]
    worker_request: str
    last_answer: str
