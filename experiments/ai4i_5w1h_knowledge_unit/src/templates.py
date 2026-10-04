"""Answer templates (the *T* component) for the 5W1H knowledge units.

Each handler in :mod:`ku_handlers` has a default template keyed by its name; a
question may override the template via ``template_id``. Templates are plain
format strings rendered with the handler's slot dictionary, which is how the
framework constrains output and removes free-form hallucination.
"""
from __future__ import annotations

# Default category templates (one sentence shape per 5W1H category).
DEFAULT_BY_CATEGORY = {
    "What": "The {metric} is {value} {unit}.",
    "Why": "Reason: {reason}; evidence: {evidence}.",
    "When": "Current stage: {stage} (value {value}, critical {threshold}).",
    "Where": "Anomaly source: {component}, based on {feature} deviation {deviation}.",
    "Who": "Impact: {process}; requires {responder} attention (severity {severity}).",
    "How": "Recommended {action}: adjust {parameter} to {suggested}.",
}

# Handler-specific templates (override the category default where the slots differ).
TEMPLATES = {
    # WHAT
    "h_machine_failure": "The sample is labelled as {label} (machine failure = {failed}).",
    "h_fault_mode": "The active fault mode is {fault} ({fault_desc}); active modes: {n_modes}.",
    "h_param_value": "The {metric} is {value} {unit}.",
    "h_param_mean": "The dataset mean {metric} is {value} {unit}.",
    "h_param_range": "The {metric} ranges from {min} to {max} {unit}.",
    "h_product_type": "Product type is {type} (quality variant: {quality}).",
    "h_model_prediction": "Model {model} predicts: {predicted_label}.",
    "h_model_confidence": "Model {model} failure probability: {probability}.",
    "h_failure_rate_dataset": "Dataset failure rate is {rate_pct}% ({n_failures}/{n_total}).",
    "h_overall_failure_rate": "Overall failure rate is {rate_pct}%.",
    "h_model_quality": "Model {model} {metric}: {value}.",
    # WHY
    "h_why_pwf": "Why PWF: {reason}; evidence: {evidence}.",
    "h_why_hdf": "Why HDF: {reason}; evidence: {evidence}.",
    "h_why_osf": "Why OSF: {reason}; evidence: {evidence}.",
    "h_why_twf": "Why TWF: {reason}; evidence: {evidence}.",
    "h_why_failure": "Primary cause: {reason}; evidence: {evidence}.",
    "h_why_model_predicts": "Why model {model} decides so: {reason}; evidence: {evidence}.",
    "h_why_no_failure": "{reason}; evidence: {evidence}.",
    "h_why_param_deviation": "{param} is {value} ({reason} vs mean {mean}).",
    # WHEN
    "h_wear_stage": "Current stage: {stage} (value {value} min, critical {threshold}); {action}.",
    "h_thermal_risk": "Heat-dissipation risk: {risk} (temp diff {value}K vs critical {threshold}K).",
    "h_power_threshold": "Power status: {status} (value {value}W, safe band {threshold}).",
    "h_overstrain_threshold": "Overstrain status: {status} (value {value} vs limit {threshold} for type {type}).",
    "h_intervention_needed": "Intervention needed: {needed} (urgency {urgency}).",
    "h_tool_replacement_due": "Tool replacement due: {due} (wear {wear}min vs threshold {threshold}min).",
    # WHERE
    "h_locate_anomaly": "Anomaly source: {component}, based on {feature} ({deviation}).",
    "h_locate_component": "Feature {feature} maps to component {component} (value {value} {unit}).",
    "h_fault_component": "Fault mode {mode} localizes to component {component}.",
    "h_component_status": "Component {component} status: {status} (representative value {value}).",
    # WHO
    "h_fault_responder": "Required responder: {responder} (severity {severity}, mode {mode}).",
    "h_affected_process": "Affected process: {process}.",
    "h_severity_level": "Severity: {severity}; responder: {responder}; mode: {mode}.",
    # HOW
    "h_recommended_action": "Recommended action ({mode}): {action}; adjust {parameter} to {suggested}.",
    "h_action_for_mode": "For {mode}: {action} (component {component}, responder {responder}).",
    "h_parameter_adjustment": "Parameter {parameter}: current {current} {unit}; suggested: {suggested}.",
    "h_prevention": "Preventive measure: {measure}.",
    # extra (statistical / model-level)
    "h_param_std": "The dataset standard deviation of {metric} is {value} {unit}.",
    "h_param_deviation": "{metric} = {value} {unit} (mean {mean}, std {std}, z={z}).",
    "h_mode_base_rate": "Base rate of {mode} ({desc}) is {rate_pct}%.",
    "h_feature_importance": "In model {model}, importance of {feature} is {importance}.",
    "h_threshold_breach": "{metric} = {value}; threshold breach: {breached} (limit {threshold}).",
    "h_why_not_mode": "Why not {mode}: {reason}; evidence: {evidence}.",
    "h_why_rnf": "{reason}; evidence: {evidence}.",
    "h_dataset_imbalance": "Class imbalance: failure rate {rate_pct}% (ratio {ratio}).",
    "h_role_for_component": "The {component} ({status}) is the responsibility of {role}.",
    "h_mitigation_for_mode": "Mitigation for {mode} (active={active}): {mitigation} (component {component}).",
}


def render(template_id: str, category: str, slots: dict) -> str:
    """Render a template, falling back to the category default then a JSON dump."""
    tpl = TEMPLATES.get(template_id) or DEFAULT_BY_CATEGORY.get(category)
    if tpl is None:
        return str(slots)
    try:
        return tpl.format(**slots)
    except KeyError:
        # tolerant render: ignore missing placeholders rather than crash
        import string
        return string.Formatter().vformat(tpl, (), _SafeDict(slots))


class _SafeDict(dict):
    def __missing__(self, key):
        return f"<{key}>"
