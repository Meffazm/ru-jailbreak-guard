"""Streamlit multi-model comparison UI for ru-jailbreak-guard.

Three views:
- Single prediction (default — calls one ISVC, picked by query param or default)
- Three-way comparison (queries all three family ISVCs concurrently)
- Recent predictions log (in-memory ring buffer of last N predictions)

The UI talks to InferenceServices over cluster DNS by default; override via env.
"""

from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx
import streamlit as st

DEFAULT_ENDPOINTS = {
    "tfidf_logreg": "http://ru-jailbreak-tfidf-predictor.model-tfidf.svc.cluster.local",
    "lgbm_emb": "http://ru-jailbreak-lgbm-predictor.model-lgbm.svc.cluster.local",
    "rubert_ft": "http://ru-jailbreak-rubert-ft-predictor.model-rubert-ft.svc.cluster.local",
}

FAMILY_LABELS = {
    "tfidf_logreg": "TF-IDF + LogReg",
    "lgbm_emb": "LightGBM on ruBERT-emb",
    "rubert_ft": "ruBERT-tiny2 fine-tuned",
}

CHAMPION_FAMILY = "rubert_ft"  # default for single-prediction view
HISTORY_LIMIT = 50


def _endpoint(family: str) -> str:
    """Override endpoints via env vars: PREDICTOR_<FAMILY_UPPER>_URL."""
    env_key = f"PREDICTOR_{family.upper()}_URL"
    return os.environ.get(env_key, DEFAULT_ENDPOINTS[family])


@dataclass
class CallResult:
    family: str
    status: int
    body: dict
    error: str | None = None
    network_ms: float = 0.0


def _call_predictor(*, family: str, text: str, timeout_s: float = 30.0) -> CallResult:
    url = f"{_endpoint(family)}/predict"
    t0 = time.perf_counter()
    try:
        resp = httpx.post(url, json={"text": text}, timeout=timeout_s)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return CallResult(
            family=family,
            status=resp.status_code,
            body=resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {},
            network_ms=elapsed,
        )
    except httpx.HTTPError as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return CallResult(family=family, status=0, body={}, error=str(exc), network_ms=elapsed)


def _render_one_result(*, result: CallResult, container) -> None:
    name = FAMILY_LABELS.get(result.family, result.family)
    with container:
        st.subheader(name)
        if result.error:
            st.error(f"Network error: {result.error}")
            return
        if result.status != 200:
            st.error(f"HTTP {result.status}")
            return
        body = result.body
        label = body.get("label", "?")
        conf = float(body.get("confidence", 0.0))
        latency_ms = float(body.get("latency_ms", 0.0))
        version = body.get("model_version", "?")
        data_v = body.get("data_version", "?")

        if label == "jailbreak":
            st.markdown(f"### :red[{label}]  ({conf:.2%})")
        else:
            st.markdown(f"### :green[{label}]  ({conf:.2%})")
        st.progress(min(max(conf, 0.0), 1.0))
        cols = st.columns(3)
        cols[0].metric("Latency", f"{latency_ms:.1f} ms")
        cols[1].metric("Network", f"{result.network_ms:.1f} ms")
        cols[2].metric("Model v", f"v{version}")
        st.caption(f"data_version: `{data_v}`")


def _add_to_history(text: str, results: list[CallResult]) -> None:
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=HISTORY_LIMIT)
    rows = []
    for r in results:
        body = r.body
        rows.append(
            {
                "ts": time.time(),
                "text": text[:80] + ("..." if len(text) > 80 else ""),
                "family": r.family,
                "label": body.get("label", "?") if r.error is None else "ERR",
                "confidence": float(body.get("confidence", 0.0)) if r.error is None else 0.0,
                "latency_ms": float(body.get("latency_ms", 0.0)) if r.error is None else 0.0,
                "version": body.get("model_version", "?") if r.error is None else "?",
            }
        )
    st.session_state.history.appendleft(rows)


def _view_single(text: str) -> None:
    family = st.session_state.get("single_family", CHAMPION_FAMILY)
    container = st.container()
    with st.spinner(f"Calling {FAMILY_LABELS[family]}..."):
        result = _call_predictor(family=family, text=text)
    _render_one_result(result=result, container=container)
    _add_to_history(text, [result])


def _view_comparison(text: str) -> None:
    cols = st.columns(len(DEFAULT_ENDPOINTS))
    families = list(DEFAULT_ENDPOINTS.keys())
    placeholders = {family: cols[i].empty() for i, family in enumerate(families)}
    for i, family in enumerate(families):
        cols[i].markdown(f"**{FAMILY_LABELS[family]}**")
        cols[i].info("calling...")

    results: dict[str, CallResult] = {}
    with ThreadPoolExecutor(max_workers=len(families)) as pool:
        futures = {pool.submit(_call_predictor, family=f, text=text): f for f in families}
        for fut in as_completed(futures):
            family = futures[fut]
            results[family] = fut.result()
            placeholders[family].empty()
            with placeholders[family].container():
                _render_one_result(result=results[family], container=st.container())

    # Agreement summary
    labels: list[str] = [
        str(r.body.get("label", ""))
        for r in results.values()
        if r.error is None and r.status == 200 and r.body.get("label") is not None
    ]
    if labels and len(set(labels)) == 1:
        st.success(f"All families agree: **{labels[0]}**")
    elif labels:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        majority = max(counts.items(), key=lambda kv: kv[1])[0]
        st.warning(f"Disagreement: {counts}. Majority: **{majority}**.")

    _add_to_history(text, list(results.values()))


def _view_history() -> None:
    history = st.session_state.get("history")
    if not history:
        st.info("No predictions yet.")
        return
    flat: list[dict] = []
    for batch in history:
        flat.extend(batch)
    st.dataframe(flat, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="ru-jailbreak-guard", page_icon=":lock:", layout="wide")
    st.title(":lock:  ru-jailbreak-guard")
    st.caption("Russian-language jailbreak classifier — three model families served via KServe.")

    with st.sidebar:
        st.header("Mode")
        view = st.radio(
            label="View",
            options=["Single (champion)", "Compare all 3", "Recent predictions"],
            label_visibility="collapsed",
        )
        if view == "Single (champion)":
            family = st.selectbox(
                "Model family",
                options=list(DEFAULT_ENDPOINTS.keys()),
                format_func=lambda f: FAMILY_LABELS[f],
                index=list(DEFAULT_ENDPOINTS.keys()).index(CHAMPION_FAMILY),
            )
            st.session_state.single_family = family

        st.divider()
        st.markdown("**Endpoints**")
        for f in DEFAULT_ENDPOINTS:
            st.caption(f"`{f}`: `{_endpoint(f)}`")

    if view == "Recent predictions":
        _view_history()
        return

    text = st.text_area(
        "Input text (Russian)",
        placeholder="Введите текст для анализа...",
        height=100,
    )
    submit = st.button("Predict", type="primary", disabled=not text.strip())
    if not submit:
        return

    if view == "Single (champion)":
        _view_single(text)
    else:
        _view_comparison(text)


if __name__ == "__main__":
    main()
