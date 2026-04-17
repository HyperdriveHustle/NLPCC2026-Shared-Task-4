"""Stub reporting/fund_arena module - upstream repo missing this file."""


def build_scenario_report(scenario: str, top_k: int = 10):
    return {"scenario": scenario, "top_k": top_k, "data": []}


def get_available_scenarios(top_k: int | None = None):
    return {"scenarios": ["default"], "top_k": top_k}


def warm_up_report_cache():
    pass
