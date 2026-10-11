"""Independent analytical validation of deterministic scalar linear dynamics."""
from dataclasses import asdict, dataclass
import math
import platform
import statistics
import time

import torch


@dataclass(frozen=True)
class Case:
    name: str
    rate: float
    initial: float = 1.0
    target: float = 0.0
    horizon: float = 1.0


@dataclass(frozen=True)
class Config:
    solver: str
    steps: int


def integrate(rate, initial, horizon, config):
    if config.steps <= 0 or config.solver not in {"euler", "rk4"}:
        raise ValueError("Require positive steps and euler or rk4")
    z = initial
    h = horizon / config.steps
    for _ in range(config.steps):
        k1 = rate * z
        if config.solver == "euler":
            z = z + h * k1
        else:
            k2 = rate * (z + h * k1 / 2)
            k3 = rate * (z + h * k2 / 2)
            k4 = rate * (z + h * k3)
            z = z + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return z


def analytical(case):
    state = case.initial * math.exp(case.rate * case.horizon)
    # Objective: 0.5 * (z(T) - target)**2; parameter: rate.
    gradient = (state - case.target) * case.horizon * state
    return state, gradient


def execute(case, config):
    rate = torch.tensor(case.rate, dtype=torch.float64, requires_grad=True)
    state = integrate(rate, torch.tensor(case.initial, dtype=torch.float64),
                      case.horizon, config)
    gradient, = torch.autograd.grad(0.5 * (state - case.target)**2, rate)
    return state.item(), gradient.item()


def check(case, config):
    state, gradient = execute(case, config)
    ref_state, ref_grad = analytical(case)
    finite = all(math.isfinite(x) for x in (state, gradient))
    se, ge = abs(state-ref_state), abs(gradient-ref_grad)
    state_limit = 1e-6 + 1e-3 * abs(ref_state)
    gradient_limit = 1e-7 + 1e-2 * abs(ref_grad)
    near_zero = abs(ref_grad) < 1e-10
    cosine = None if near_zero or abs(gradient) < 1e-10 else math.copysign(1, gradient*ref_grad)
    reasons = []
    if not finite:
        reasons.append("nonfinite_state_or_gradient")
    if se > state_limit:
        reasons.append("state_tolerance_exceeded")
    if ge > gradient_limit:
        reasons.append("gradient_tolerance_exceeded")
    result = {"case": asdict(case), "state": state, "gradient": gradient,
            "reference_state": ref_state, "reference_gradient": ref_grad,
            "state_absolute_error": se, "gradient_absolute_error": ge,
            "gradient_normalized_error": ge/max(abs(ref_grad), 1e-10) if finite else None,
            "gradient_cosine": cosine, "near_zero_reference_gradient": near_zero,
            "state_limit": state_limit, "gradient_limit": gradient_limit,
            "passed": finite and not reasons, "failures": reasons}
    # Strict JSON: nonfinite measurements become null; failure reasons persist.
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v
            for k, v in result.items()}


def run(cases=None, configs=None):
    cases = cases or [Case("stable", -1.0), Case("stiff_decay", -10.0),
                      Case("zero_gradient", -1.0, initial=0.0)]
    configs = configs or [Config(s, n) for s in ("euler", "rk4") for n in (2, 16, 128)]
    rows = []
    for config in configs:
        checks = [check(case, config) for case in cases]
        for _ in range(2):
            for case in cases:
                execute(case, config)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            for case in cases:
                execute(case, config)
            timings.append(1000*(time.perf_counter()-start))
        rows.append({"configuration": asdict(config), "checks": checks,
                     "passed": all(c["passed"] for c in checks),
                     "latency_ms_median": statistics.median(timings),
                     "latency_ms_min": min(timings), "latency_ms_max": max(timings)})
    passing = [r for r in rows if r["passed"]]
    selected = min(passing, key=lambda r:r["latency_ms_median"])["configuration"] if passing else None
    return {"schema_version": 1, "manifest": {"python": platform.python_version(),
            "pytorch": torch.__version__, "platform": platform.platform(),
            "device": "cpu", "dtype": "float64", "torch_threads": torch.get_num_threads(),
            "objective": "0.5*(final_state-target)^2", "gradient_parameter": "rate",
            "timing": "7 repeats, 2 warmups; complete forward+backward across all cases",
            "limitations": "Exploratory CPU timings; scalar gradients; final-state validation only"},
            "tolerances": {"state_atol": 1e-6, "state_rtol": 1e-3,
                           "gradient_atol": 1e-7, "gradient_rtol": 1e-2,
                           "small_gradient_threshold": 1e-10},
            "results": rows, "selected_configuration": selected,
            "recommendation_scope": "Only evaluated cases, objective and environment; timing ties not resolved"}
