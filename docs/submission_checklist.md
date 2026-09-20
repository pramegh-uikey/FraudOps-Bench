# IEEE Access submission checklist

Tracks what's left before `paper/latex/fraudops_bench.tex` goes to the
IEEE Access portal. `main` is otherwise submission-ready (compiles
clean, all numbers verified against `outputs/`/`docs/methodology_log.md`)
-- everything below is account-level or author-judgment work, not
anything I can finish unilaterally.

## Open items

- [ ] **References are preprint-heavy.** 9 of 24 entries are arXiv-only
  and only `chow1970` carries a DOI. Where a peer-reviewed version now
  exists (τ-bench, AutoGen, SOP-Bench are the likely candidates),
  IEEE prefers it, and IEEE Access asks for DOIs where available. Not
  changed in the fix round: verifying current publication venues needs
  a literature check I can't do offline, and guessing a venue is worse
  than citing the preprint honestly.
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

### Pre-submission review round (branch `paper-review-fixes`, 2026-09-19)

A full verification pass recomputed every reported number from
`outputs/` -- Tables 3/4/7 from the per-arm `*_metrics.csv`, Table 2
from the calibration `rep1`/`rep2` runs, Section V-D's discretization
counts from `holdout/agentic_api_parsed.jsonl`, faithfulness from the
`*_faithfulness.csv` and `*_judge_faithfulness.csv` files, and split
disjointness from the case files. Everything matched except two cost
figures. Changes made:

- **Balanced-sampling disclosure (content, was missing entirely).**
  Every split is 50/50 fraud by construction while IEEE-CIS runs at
  3.50% (7.96% in the device-bearing pool cases are drawn from). Added
  a sampling paragraph to Section III-A and a Limitations bullet
  stating plainly that no absolute number here is a deployment
  estimate.
- **Case-eligibility filter disclosure.** `build_pool()` restricts
  sampling to transactions carrying device information -- 23.9% of the
  dataset. Previously invisible in the paper; now stated in
  Section III-A with its own Limitations bullet.
- **New Section III-C, implementation details.** Model IDs, the
  provider-default decoding constraint, token caps, retry policy, and a
  full `classical_ml` specification (42 features, `class_weight`,
  `random_state`, training-set construction, the unreported logreg
  variant).
- **New metrics formalization in Section III-D.** Coverage, selective
  risk, coverage-weighted score, and AURC as equations (1)-(4); the
  paper previously quantified all four in prose only.
- **Section V-B narrative order corrected.** The text claimed the
  repeat-run variance check preceded the root-cause diagnosis; the
  methodology log dates the diagnosis to 2026-08-16 and the repeat runs
  to 2026-08-18, and Table 2 is scored under the *corrected* band. Now
  says so directly rather than implying a tidier order than happened.
- **Figure 2 caption.** It plots accuracy but was captioned as a
  risk-coverage curve quoting AURC; now states the axis relationship
  explicitly. Also corrected "all five arms" to name why
  `direct_control` is absent.
- **Cost figures reconciled to the shipped data.** Section VII said the
  ablation cost \$61.87 and saved \$10.04 on `linear_api`; summing
  `cost_usd` in the supplementary CSVs gives \$61.77 and \$10.14. The
  paper now matches what a reviewer would compute. (\$92.84, \$63.88,
  \$26.06 and the `agentic_api` \$6.85 all verified exact.)
- **Seven typographic defects** where `\code{}`/`\url` swallowed a
  space or a line-ending `/` or `-` injected one -- most visibly
  `--override-self-consistencyfalse` in Section VI. All fixed and
  confirmed in the rebuilt PDF.
- Smaller fixes: faithfulness range corrected to 98.7-99.3% (pooled
  `holdout_v2/linear_api` is 98.72%); abstract "all four LLM arms" ->
  "all four evidence-bearing arms" (`direct_control` is an LLM arm too
  and was not faithfulness-scored); McNemar table now defines *b* and
  *c*; Section I cites the dataset rather than FDB for the dataset;
  "900 successful API calls" -> "900 case runs" (`agentic_api` issues
  several model calls per case); consistent `$n$` styling in captions;
  `\texorpdfstring` on three subsection titles, clearing all six
  hyperref PDF-bookmark warnings.

### Author photos added (2026-09-20)

Both `IEEEbiography` entries now carry headshots supplied by the
authors, replacing the `[\mbox{}]` placeholders:
`images/ayushi_ambilkar.png` and `images/pramegh_uikey.png`, included at
`width=1in,height=1.25in,clip,keepaspectratio`.

Each source image was cropped to IEEE's 4:5 biography box rather than
letterboxed, trimming the dead headroom above the subject (both
originals were taller than 4:5, so a straight fit would have left white
gaps and inconsistent head sizes between the two). Ayushi's also needed
a light 60px side trim to bring its headroom in line with Pramegh's;
the result is 6.4% and 7.7% headroom respectively, so the two read as a
matched pair. Both were then resampled to 600x750 px -- exactly 600 dpi
at the printed size, twice IEEE's 300 dpi floor. The unmodified
originals are still in `~/Downloads`; re-crop from those if you want
different framing.

This takes the PDF from 172 KB to 1.2 MB, which is immaterial for the
portal's limits.

### Checked against IEEE Access's current guidelines (2026-09-20)

- **Page count is a non-issue; an earlier note in this file was wrong.**
  It claimed the APC covers 10 pages with per-page overlength charges.
  It does not. IEEE Access charges a flat $2,160 per article and states
  plainly: "There is no page limit for articles and therefore no
  over-length article charge." They only recommend staying under 20
  pages for readability. At 13 pages there is nothing to pay and no
  reason to trim. The 10-page/overlength rule belongs to other IEEE
  Transactions, not Access. Source:
  https://ieeeaccess.ieee.org/about-ieee-access/article-processing-charges/
- **Line numbers: not required, and not available anyway.** Nothing in
  the Submission Guidelines or Preparing Your Article pages mentions
  line numbers. More to the point, the `lineno` option is dead code in
  our `ieeeaccess.cls`: `\iflineno` is declared (line 12), set by
  `\DeclareOption` (lines 30/32) and switched on by
  `\ExecuteOptions{...,lineno,...}` (line 44), but the flag is never
  tested anywhere in the class and the `lineno` package is never
  loaded. Test-compiled with `\documentclass[lineno]{ieeeaccess}`: the
  extracted text is byte-identical to the current build. Do not bother.
- **The first-page furniture is correct as-is.** The "RESEARCH ARTICLE"
  banner, the real received/accepted/published dates, the assigned DOI,
  the journal page number and the associate-editor line in a published
  IEEE Access PDF are all added by IEEE production after acceptance.
  `ieeeaccess.cls` has no command for the banner at all. The
  `\history{... xxxx 00, 0000 ...}` and `\doi{10.1109/ACCESS.2017.DOI}`
  placeholders are exactly what the official template ships with and
  are the correct state to submit in.

### Supplementary material rebuilt

`paper/supplementary_holdout_v2.zip` had picked up `linear_retrieval`
and `agentic_retrieval` artifacts from the `retrieval-exemplars`
branch's working tree -- arms that appear nowhere in the manuscript.
Worse, the bundled `stats_report.md` (which Section V-F explicitly
directs reviewers to) and `risk_coverage_curves.png` were the 7-arm
versions: they contradicted Figure 2 and showed the undisclosed arms
scoring *better* AURC (0.0982 / 0.0896) than either reported Claude arm
(0.1080 / 0.1117). Both reports and the figure were regenerated from
the five paper arms and the zip rebuilt with the retrieval files
excluded (36 entries, verified). The 7-arm originals were preserved
before regeneration -- they belong on `retrieval-exemplars`, not in the
submission bundle.

### Earlier rounds

- Table numbering, figure-caption accuracy, bullet-dot alignment, and
  Figure 1's arrow/label overlaps were all fixed and pushed to `main`
  earlier (see recent commit history).
- Title changed to *"...and a Calibration Pitfall in Selective
  Prediction with LLM Confidence Scores"* (from "...Worth Knowing
  About").
- The stray, unrelated PDF in `paper/` (`High-Gain_Circularly_...pdf`)
  is still sitting there untracked -- your call whether to delete it,
  move it, or leave it; not blocking submission either way.
