# IEEE Access submission checklist

Tracks what's left before `paper/latex/fraudops_bench.tex` goes to the
IEEE Access portal. `main` is otherwise submission-ready (compiles
clean, all numbers verified against `outputs/`/`docs/methodology_log.md`)
-- everything below is account-level or author-judgment work, not
anything I can finish unilaterally.

## Open items

- [ ] **Author photos.** Both `IEEEbiography` entries
  (`paper/latex/fraudops_bench.tex`, ~line 1331 and ~line 1341) currently
  use `[\mbox{}]` as a placeholder for the optional photo argument (no
  photographs were available when the biographies were written). Once
  you have headshots: drop the image files in `paper/latex/images/`,
  replace each `[\mbox{}]` with
  `[\includegraphics[width=1in,height=1.25in]{images/<file>}]`, and
  recompile to confirm sizing/cropping looks right.
- [ ] **ORCID iDs, both authors.** IEEE's own Submission Checklist item 4
  only strictly requires the *corresponding* author's (Ayushi's) ORCID
  in the submission portal -- the TODO comment in the `.tex`
  (~line 67) currently only mentions her. You've now said you want both
  authors registered, which is good practice beyond the strict
  requirement. Each author registers separately at orcid.org (an
  account-level action, not a document field); Ayushi's goes in the
  portal per the checklist, and if you want Pramegh's iD reflected in
  the manuscript too, IEEE Access's `\author`/`\address` commands support
  an optional ORCID field -- flag it here once both iDs exist and I'll
  wire it in.
- [ ] **Acknowledgments revisions -- specifics TBD.** You mentioned
  "some minor revisions" without saying what yet. Current text
  (`fraudops_bench.tex`, `\section*{Acknowledgments}`, ~line 1189)
  discloses AI-assisted drafting of Introduction/Related
  Work/Discussion/Conclusion, citation verification, and integrating
  already-existing experimental data into text -- with all experimental
  design, code, data collection, and statistical analysis credited to
  the authors. Come back to this with what you want changed and I'll
  make the edit.

## Already resolved (for reference, not action items)

- Table numbering, figure-caption accuracy, bullet-dot alignment, and
  Figure 1's arrow/label overlaps were all fixed and pushed to `main`
  earlier (see recent commit history).
- Title changed to *"...and a Calibration Pitfall in Selective
  Prediction with LLM Confidence Scores"* (from "...Worth Knowing
  About").
- The stray, unrelated PDF in `paper/` (`High-Gain_Circularly_...pdf`)
  is still sitting there untracked -- your call whether to delete it,
  move it, or leave it; not blocking submission either way.
