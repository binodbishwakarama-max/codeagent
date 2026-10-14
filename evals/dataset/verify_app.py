"""Streamlit Human Verification App for PR Gold Findings.

Enables human curators to review candidate gold findings alongside the unified diff:
- Accept: marks verified=True
- Edit: modifies category, severity, line, or description, then marks verified=True
- Reject: removes the finding from the gold set
- Wrong Line: adjusts the line number and marks verified=True

Usage:
    streamlit run evals/dataset/verify_app.py
    # or
    python -m streamlit run evals/dataset/verify_app.py -- --file data/final/test.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from evals.dataset.labeling import load_jsonl, save_jsonl
from evals.dataset.schema import FindingCategory, FindingSeverity, PRExample


def get_default_file_path() -> Path:
    """Parse optional --file argument from CLI or default to data/final/test.jsonl."""
    default_path = Path("data/final/test.jsonl")
    if "--file" in sys.argv:
        try:
            idx = sys.argv.index("--file")
            if idx + 1 < len(sys.argv):
                default_path = Path(sys.argv[idx + 1])
        except Exception:
            pass
    elif not default_path.exists() and Path("data/raw/prs.jsonl").exists():
        default_path = Path("data/raw/prs.jsonl")
    return default_path


def load_dataset_cached(path_str: str) -> list[PRExample]:
    """Load dataset from JSONL path."""
    path = Path(path_str)
    if not path.exists():
        return []
    return load_jsonl(path)


def save_current_dataset(examples: list[PRExample], path: Path) -> None:
    """Atomically write modified examples to the JSONL file."""
    save_jsonl(examples, path)


def is_pr_fully_verified(example: PRExample) -> bool:
    """Return True if all gold findings in the PR are verified (or PR is clean)."""
    if not example.gold_findings:
        return True
    return all(f.verified for f in example.gold_findings)


def main() -> None:
    st.set_page_config(
        page_title="Prism Review — Benchmark Verifier",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -----------------------------------------------------------------------
    # Sidebar: File Selection & Overall Verification Metrics
    # -----------------------------------------------------------------------
    st.sidebar.title("🔍 Benchmark Verifier")
    st.sidebar.markdown(
        "Human verification portal for establishing ground-truth evaluation datasets."
    )

    default_file = get_default_file_path()
    file_path_str = st.sidebar.text_input("Dataset File (JSONL)", str(default_file))
    dataset_path = Path(file_path_str)

    # Initialize or reload session state
    if "dataset_path" not in st.session_state or st.session_state.dataset_path != file_path_str:
        st.session_state.dataset_path = file_path_str
        st.session_state.examples = load_dataset_cached(file_path_str)
        st.session_state.current_idx = 0
        st.session_state.editing_finding = None

    if st.sidebar.button("🔄 Reload File from Disk"):
        st.session_state.examples = load_dataset_cached(file_path_str)
        st.session_state.editing_finding = None
        st.sidebar.success("Reloaded!")

    examples: list[PRExample] = st.session_state.examples

    if not examples:
        st.warning(f"No examples found in `{dataset_path}`. Please verify path.")
        st.stop()

    total_prs = len(examples)
    total_findings = sum(len(ex.gold_findings) for ex in examples)
    verified_findings = sum(1 for ex in examples for f in ex.gold_findings if f.verified)
    verified_prs = sum(1 for ex in examples if is_pr_fully_verified(ex))

    finding_pct = (100.0 * verified_findings / max(1, total_findings)) if total_findings else 100.0
    pr_pct = (100.0 * verified_prs / max(1, total_prs)) if total_prs else 100.0

    st.sidebar.divider()
    st.sidebar.subheader("📊 Verification Progress")
    st.sidebar.metric(
        "Findings Verified", f"{verified_findings} / {total_findings}", f"{finding_pct:.1f}%"
    )
    st.sidebar.metric("PRs Verified", f"{verified_prs} / {total_prs}", f"{pr_pct:.1f}%")
    st.sidebar.progress(finding_pct / 100.0)

    if finding_pct >= 100.0:
        st.sidebar.success("🎉 Target Achieved: 100% Verified!")

    # -----------------------------------------------------------------------
    # Filter & Navigation
    # -----------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("PR Navigation")

    filter_mode = st.sidebar.radio(
        "Filter View",
        ["All PRs", "Unverified PRs Only", "Verified PRs Only", "Clean PRs (0 findings)"],
    )

    indices: list[int] = []
    for idx, ex in enumerate(examples):
        if filter_mode == "All PRs":
            indices.append(idx)
        elif filter_mode == "Unverified PRs Only":
            if not is_pr_fully_verified(ex):
                indices.append(idx)
        elif filter_mode == "Verified PRs Only":
            if is_pr_fully_verified(ex):
                indices.append(idx)
        elif filter_mode == "Clean PRs (0 findings)":
            if not ex.gold_findings:
                indices.append(idx)

    if not indices:
        st.info(f"No PRs match filter: **{filter_mode}**")
        st.stop()

    # Ensure current_idx is valid
    if st.session_state.current_idx not in indices:
        st.session_state.current_idx = indices[0]

    current_pos = indices.index(st.session_state.current_idx)

    col_nav1, col_nav2 = st.sidebar.columns(2)
    if col_nav1.button("⬅️ Previous", disabled=(current_pos == 0)):
        st.session_state.current_idx = indices[current_pos - 1]
        st.session_state.editing_finding = None
        st.rerun()

    if col_nav2.button("Next ➡️", disabled=(current_pos >= len(indices) - 1)):
        st.session_state.current_idx = indices[current_pos + 1]
        st.session_state.editing_finding = None
        st.rerun()

    selected_pos = st.sidebar.selectbox(
        "Jump to PR",
        options=range(len(indices)),
        format_func=lambda p: (
            f"[{p + 1}/{len(indices)}] "
            f"{examples[indices[p]].repo} #{examples[indices[p]].pr_number}"
        ),
        index=current_pos,
    )
    if selected_pos != current_pos:
        st.session_state.current_idx = indices[selected_pos]
        st.session_state.editing_finding = None
        st.rerun()

    # -----------------------------------------------------------------------
    # Main View: Current PR
    # -----------------------------------------------------------------------
    cur_idx = st.session_state.current_idx
    example = examples[cur_idx]

    # PR Header
    h_col1, h_col2, h_col3, h_col4 = st.columns([4, 2, 2, 2])
    with h_col1:
        st.markdown(f"### `{example.repo}` — PR #{example.pr_number}")
        st.markdown(f"**Title:** [{example.title}]({example.pr_url})")
    with h_col2:
        st.metric("Language", example.language.capitalize())
    with h_col3:
        st.metric("Changed Lines", f"{example.changed_lines}")
    with h_col4:
        split_badge = (example.split or "unassigned").upper()
        st.metric("Split", split_badge)

    if example.body:
        with st.expander("PR Description"):
            st.markdown(example.body)

    # Unified Diff
    with st.expander("📄 View Unified Diff", expanded=True):
        st.code(example.diff, language="diff")

    st.divider()

    # -----------------------------------------------------------------------
    # Findings Verification Section
    # -----------------------------------------------------------------------
    st.subheader(f"Candidate Gold Findings ({len(example.gold_findings)})")

    if not example.gold_findings:
        st.info("ℹ️ This PR has zero findings (Clean PR false-positive baseline).")
        st.stop()

    # Bulk verify button
    if any(not f.verified for f in example.gold_findings):
        if st.button("✅ Accept & Verify All Findings in this PR", type="primary"):
            updated_findings = [
                f.model_copy(update={"verified": True}) for f in example.gold_findings
            ]
            examples[cur_idx] = example.model_copy(update={"gold_findings": updated_findings})
            save_current_dataset(examples, dataset_path)
            st.success("All findings marked verified!")
            st.rerun()

    # Render each finding
    for f_idx, finding in enumerate(example.gold_findings):
        status_label = "✅ Verified" if finding.verified else "⚠️ Pending Review"
        status_color = "green" if finding.verified else "orange"

        with st.container(border=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns([5, 2, 2, 2])
            with f_col1:
                st.markdown(f"**Finding #{f_idx + 1}** — `:{status_color}[{status_label}]`")
                st.markdown(f"📁 **File:** `{finding.path}:{finding.line}`")
            with f_col2:
                st.markdown(f"🏷️ **Category:** `{finding.category.value}`")
            with f_col3:
                st.markdown(f"⚡ **Severity:** `{finding.severity.value}`")
            with f_col4:
                st.markdown(f"👤 **Reviewer:** `{finding.reviewer or 'anonymous'}`")

            st.markdown(f"**Description:** {finding.description}")

            # Check if this finding is being edited
            is_editing = st.session_state.editing_finding == f_idx

            if is_editing:
                st.divider()
                st.markdown("#### ✏️ Edit Finding")
                with st.form(key=f"edit_form_{cur_idx}_{f_idx}"):
                    e_col1, e_col2, e_col3 = st.columns([3, 2, 2])
                    with e_col1:
                        new_path = st.text_input("Path", value=finding.path)
                    with e_col2:
                        cat_list = [c.value for c in FindingCategory]
                        cur_cat_idx = (
                            cat_list.index(finding.category.value)
                            if finding.category.value in cat_list
                            else 0
                        )
                        new_cat_str = st.selectbox("Category", cat_list, index=cur_cat_idx)
                    with e_col3:
                        sev_list = [s.value for s in FindingSeverity]
                        cur_sev_idx = (
                            sev_list.index(finding.severity.value)
                            if finding.severity.value in sev_list
                            else 0
                        )
                        new_sev_str = st.selectbox("Severity", sev_list, index=cur_sev_idx)

                    new_line = st.number_input("Line", min_value=1, value=finding.line)
                    new_desc = st.text_area("Description", value=finding.description)

                    sub_c1, sub_c2 = st.columns(2)
                    if sub_c1.form_submit_button("💾 Save & Verify", type="primary"):
                        edited_finding = finding.model_copy(
                            update={
                                "path": new_path.strip(),
                                "line": int(new_line),
                                "category": FindingCategory(new_cat_str),
                                "severity": FindingSeverity(new_sev_str),
                                "description": new_desc.strip(),
                                "verified": True,
                            }
                        )
                        updated_findings = list(example.gold_findings)
                        updated_findings[f_idx] = edited_finding
                        examples[cur_idx] = example.model_copy(
                            update={"gold_findings": updated_findings}
                        )
                        save_current_dataset(examples, dataset_path)
                        st.session_state.editing_finding = None
                        st.toast("Finding updated and verified!")
                        st.rerun()

                    if sub_c2.form_submit_button("Cancel"):
                        st.session_state.editing_finding = None
                        st.rerun()

            else:
                # Action Buttons Row
                btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

                with btn_c1:
                    if st.button(
                        "✅ Accept", key=f"accept_{cur_idx}_{f_idx}", disabled=finding.verified
                    ):
                        updated_findings = list(example.gold_findings)
                        updated_findings[f_idx] = finding.model_copy(update={"verified": True})
                        examples[cur_idx] = example.model_copy(
                            update={"gold_findings": updated_findings}
                        )
                        save_current_dataset(examples, dataset_path)
                        st.toast(f"Finding #{f_idx + 1} verified!")
                        st.rerun()

                with btn_c2:
                    if st.button("✏️ Edit", key=f"edit_btn_{cur_idx}_{f_idx}"):
                        st.session_state.editing_finding = f_idx
                        st.rerun()

                with btn_c3:
                    if st.button("🗑️ Reject", key=f"reject_{cur_idx}_{f_idx}"):
                        updated_findings = [
                            f for i, f in enumerate(example.gold_findings) if i != f_idx
                        ]
                        examples[cur_idx] = example.model_copy(
                            update={
                                "gold_findings": updated_findings,
                                "is_clean": len(updated_findings) == 0,
                            }
                        )
                        save_current_dataset(examples, dataset_path)
                        st.toast(f"Finding #{f_idx + 1} removed.")
                        st.rerun()

                with btn_c4:
                    with st.popover("📍 Wrong Line"):
                        correct_line = st.number_input(
                            "Correct Line #",
                            min_value=1,
                            value=finding.line,
                            key=f"line_inp_{cur_idx}_{f_idx}",
                        )
                        if st.button("Apply Line & Verify", key=f"line_btn_{cur_idx}_{f_idx}"):
                            updated_findings = list(example.gold_findings)
                            updated_findings[f_idx] = finding.model_copy(
                                update={"line": int(correct_line), "verified": True}
                            )
                            examples[cur_idx] = example.model_copy(
                                update={"gold_findings": updated_findings}
                            )
                            save_current_dataset(examples, dataset_path)
                            st.toast(f"Updated line to {correct_line} and verified!")
                            st.rerun()


if __name__ == "__main__":
    main()
