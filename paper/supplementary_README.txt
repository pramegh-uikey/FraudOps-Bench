FraudOps-Bench -- supplementary material
Per-arm run outputs for the single-use holdout_v2 evaluation set (n = 300)

These files are the raw evidence behind Tables 3-10 and Figure 2 of the
manuscript. Everything here was produced by the code at
github.com/pramegh-uikey/FraudOps-Bench under the frozen-methodology
manifest described in Section III-E.

ARMS
  direct_control   zero-evidence control (alert summary only)
  linear_api       single-shot, all six tools' output pre-attached
  agentic_api      LangGraph agent, chooses its own tool calls
  classical_ml     HistGradientBoostingClassifier, no LLM
  linear_gpt       linear flow on GPT-5.6 Terra (Section V-H)
  agentic_gpt      agentic flow on GPT-5.6 Terra (Section V-H)

  linear_api and agentic_api are Claude Sonnet 5. The *_no_sc files are
  the same two arms re-run with self-consistency disabled -- the
  ablation of Section VI.

FILES, BY SUFFIX
  <arm>.jsonl                    raw model responses, one JSON per case.
                                 Absent for classical_ml, which makes no
                                 API calls.
  <arm>_parsed.jsonl             parsed disposition, fraud_probability,
                                 cited evidence, check verdicts, risk and
                                 protective indicators, token counts.
  <arm>_metrics.csv              one row per case: ground_truth_is_fraud,
                                 disposition, fraud_probability,
                                 self_consistency_triggered,
                                 predicted_is_fraud, tool/check counts,
                                 latency_ms, cost_usd. Every coverage and
                                 accuracy figure in the paper is
                                 recomputable from these columns and a
                                 band (l, u): a case is decided when
                                 fraud_probability <= l or >= u.
  <arm>_faithfulness.csv         Method 1, deterministic numeric
                                 cross-referencing (Section IV):
                                 n_claimed_numbers, n_verified,
                                 verified_rate, per case.
  <arm>_judge_faithfulness.csv   Method 2, LLM-judge semantic check
                                 (Section IV): n_misattributions and the
                                 judge's findings, 25-case sample per arm.
                                 Rows with a non-empty error column are
                                 judge-response parse failures and are
                                 excluded from the rates reported.

REPORTS AND FIGURE
  stats_report.md          bootstrap 95% CIs (10,000 resamples), pairwise
                           McNemar tests, and AURC per arm. Source of
                           Tables 5, 6 and 8.
  comparison_report.md     cross-arm comparison over the same cases.
  risk_coverage_curves.png Figure 2.

INPUTS
  evidence_packets.jsonl   the leakage-free tool evidence served to every
                           evidence-bearing arm, so tool outputs can be
                           checked against what the models actually saw.

NOTES
  The 300 holdout_v2 cases are sampled 50/50 fraud/non-fraud and only
  from transactions carrying device information. Section III-A and the
  Limitations section state what that does and does not support: these
  are discrimination results under a balanced queue, not deployment
  estimates at the 3.50% base rate of the underlying IEEE-CIS data.

  The raw IEEE-CIS competition data is not redistributed here, per the
  Kaggle competition terms.
