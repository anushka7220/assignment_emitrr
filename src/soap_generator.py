from src.soap_builders import (
    build_subjective,
    build_objective,
    build_assessment,
    build_plan
)

def generate_soap(grouped):
    subjective = build_subjective(grouped.get("Subjective", []))
    objective  = build_objective(grouped.get("Objective", []))

    assessment = build_assessment(
        grouped.get("Assessment", []),
        subjective=subjective,   # dict ✅
        objective=objective      # dict ✅
    )

    plan = build_plan(
        grouped.get("Plan", []),
        assessment=assessment,
        objective=objective
    )

    return {
        "Subjective": subjective,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }
