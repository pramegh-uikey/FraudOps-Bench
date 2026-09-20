# IEEE Access submission checklist

Tracks what's left before `paper/latex/fraudops_bench.tex` goes to the
IEEE Access portal. `main` is otherwise submission-ready (compiles
clean, all numbers verified against `outputs/`/`docs/methodology_log.md`)
-- everything below is account-level or author-judgment work, not
anything I can finish unilaterally.

## Open items

Checked against IEEE Access's official 18-item submission checklist
(https://ieeeaccess.ieee.org/guide-for-authors/submission-guidelines/,
retrieved 2026-09-20). Portal is the **IEEE Author Portal**,
https://ieee.atyponrex.com/journal/ieee-access -- not ScholarOne, which
an earlier note in this file wrongly assumed.

Already satisfied by the manuscript (nothing to do):

| # | Item | Status |
|---|---|---|
| 1 | Template, source + PDF matching, <40MB | 7.3 MB total, archive verified by clean-room compile |
| 2 | AI-generated text disclosed in Acknowledgments, citing the system | Itemized by section and level of use, cites [24] |
| 5 | All authors in both source and PDF | Both, with ORCIDs |
| 6 | Biographies for all authors below references | Both, with photos |
| 10 | Supplementary material ready | `supplementary_holdout_v2.zip` |
| 11 | Abbreviations defined at first body use | Swept 2026-09-20; AURC was the one miss, fixed |
| 15 | Video | N/A |
| 16 | Resubmission response document | N/A, first submission |
| 17 | Under 20 pages, else EiC pre-approval | 13 pages |
| 18 | No "Lena image" | N/A |

Requires an author decision or action:

- [ ] **(3) Confirm the author list is final.** Post-submission changes
  need editor approval and are rarely granted.
- [ ] **(9) Confirm not under consideration elsewhere.**
- [ ] **(12) Enter 3-10 keywords in the portal.** The eight in the
  manuscript are a separate field; the portal ones drive Associate
  Editor assignment.
- [ ] **(13) Manuscript type: "Research Article".** This is what
  produces the banner on the published first page.
- [ ] **(14) Opposed reviewers list**, if you want one.
- [ ] **Upload three files.** Built and verified 2026-09-20.
  - `paper/latex/fraudops_bench.pdf` (1.2 MB) -- the manuscript.
  - `paper/fraudops_bench_latex_source.zip` (1.1 MB) -- the LaTeX
    source: `fraudops_bench.tex`, `ieeeaccess.cls`, and six images.
    Three of those (`Logo.png`, `notaglineLogo.png`, `bullet.png`) are
    loaded by the class and named nowhere in the `.tex`; the build
    fails without them, which is the easy mistake here.
  - `paper/supplementary_holdout_v2.zip` (5.3 MB) -- per-arm run
    outputs.

  **Regenerate with `paper/build_submission.sh` after any edit to the
  `.tex` or its images.** The zip is a snapshot; edit the source
  without rerunning it and you will upload a source archive that does
  not match your PDF, which IEEE requires to match exactly. The script
  rebuilds the archive and then proves it by extracting to an empty
  directory and compiling there -- it exits non-zero if a file is
  missing.
- [ ] **Verify the portal extracted both authors** during upload
  (item 5's second half) before you finish the submission.

Note: the APC ($2,160) is charged **after acceptance**, not at
submission. IEEE Access advertises 4-6 weeks from submission to online
publication.

## Already resolved (for reference, not action items)

### ORCID iDs and Acknowledgments (2026-09-20)

- Both ORCID iDs are now in `\author`, rendered as linked iD icons via
  the `orcidlink` package (`ieeeaccess.cls` has none of its own):
  Ayushi 0009-0000-5624-7588, Pramegh 0009-0004-1052-8703. Link targets
  verified in the built PDF. **Still enter both in the submission
  portal** -- that record is what IEEE production uses for the
  published PDF.
- Ayushi's inferred Accertify end year (2026) confirmed correct by the
  authors, who also supplied her Accenture title (AI Decision Science
  Consultant). Both biographies are now complete.
- Acknowledgments rewritten for IEEE AI-disclosure compliance. The old
  text predated most of the AI-assisted writing in the manuscript and
  named only four sections. IEEE requires that "specific sections of
  the article that use AI-generated content shall be identified and
  accompanied by a brief explanation regarding the level at which the
  AI system was used." The new text itemizes three distinct levels of
  use. A sentence separating the writing assistant from the evaluated
  models was drafted and then removed at the authors' direction: the
  two uses are causally unrelated (the runs predate the drafting and
  are reproducible from the released outputs), and the paper's headline
  finding is that classical ML beats every LLM arm including Claude, so
  there is no direction in which authoring assistance could have
  flattered the evaluated model. Disclaiming it would have implied a
  concern the results do not support.

### Grammar and retraction passes (2026-09-20)

**Grammar (item 7).** Read the full rendered body end to end. Seven
real fixes, all now in the PDF:

- "an 0.85 target" -> "a 0.85 target" (Section V-E) -- wrong article
  before a numeral read aloud as "zero point...".
- "rather than surprising *the model* with a burst of wrong
  high-confidence calls" -> "rather than springing a burst of..."
  (Section V-H). The model was not the party being surprised; the
  sentence said the opposite of what it meant.
- Conclusion cited "the repeat-run stochasticity check (Section IX)".
  Section IX is Limitations; the check is Section V-B. Cross-reference
  corrected.
- Section III-A gave the same figure as both "3.50%" and "3.5%" two
  sentences apart. Now 3.50% in both.
- Five "e.g." without the comma IEEE's style manual requires -> "e.g.,".
- Three hyphens after -ly adverbs removed
  ("independently-selected", "correctly-detected",
  "differently-shaped"), which IEEE style does not hyphenate.

**Retractions (item 8).** All 24 checked, none retracted or withdrawn.

- The eight arXiv preprints (`fdb2022`, `finfrerag2025`, `faa2025`,
  `cortex2025`, `siabench2026`, `sopbench2025`, `intellagent2025`,
  `dai2026`) were queried through the arXiv API in one call: every one
  returns active, none carries a withdrawal notice, and the returned
  titles match our bibliography entries.
- `geifman2017` (NeurIPS proceedings page) and `tian2023` /
  `campos2024` (ACL Anthology / MIT Press, both fetched while adding
  DOIs) carry no retraction or erratum notice.
- `baf2022` and `amlworld2023` searched specifically for retraction
  notices: none found.
- Six entries are reports, datasets or software (`acfe2024`,
  `ukfinance2025`, `ieeecis2019`, `langgraph2024`, `crewai2023`,
  `claude2026`) and are not retractable article records.
- `chow1970`, `xiong2024`, `bfcl2025`, `taubench2024`, `autogen2023`
  turned up no retraction notice in searching, though for these the
  evidence is absence-of-notice rather than a positive check against a
  retraction database.

### References checked against published venues (2026-09-20)

Four entries had peer-reviewed versions and were updated:

- `taubench2024` -> Proc. ICLR 2025 (was arXiv:2406.12045)
- `autogen2023` -> Proc. 1st Conf. Language Modeling (COLM) 2024
  (was arXiv:2308.08155); title also corrected to the published form,
  which drops "framework"
- `campos2024` -> Trans. Assoc. Comput. Linguistics, vol. 12, pp.
  1497-1516, 2024, doi: 10.1162/tacl_a_00715 (was arXiv:2405.01976);
  first author's initials corrected from "M. M. Campos" to "M. Campos"
- `tian2023` -> EMNLP venue was already correct; added
  doi: 10.18653/v1/2023.emnlp-main.330

Checked and left as preprints because no published version was found:
`sopbench2025` (revised Feb 2026, still arXiv), `intellagent2025`,
`fdb2022`, plus the recent ones (`finfrerag2025`, `faa2025`,
`cortex2025`, `siabench2026`, `dai2026`). Eight arXiv citations remain,
which is defensible for work this recent.

DOIs: `chow1970`, `campos2024` and `tian2023` now carry them. The
remaining venues -- NeurIPS, ICLR, ICML/PMLR -- do not mint DOIs for
proceedings papers, so there are none to add.

### Abbreviation check (2026-09-20)

IEEE Access requires abbreviations to be defined at first use in the
body "even if defined in the abstract." Swept every all-caps token in
the rendered text against its first body occurrence. One real miss:
AURC was used from Section III-D onward but only spelled out in
Section V-H, long after. The expansion now sits at the first body use
and the late redundant one is gone. Everything else checked out --
LLM, ML, API, SOP, SOC, RLHF, CI, LCB, PII, RAG, AURC, IIT all defined
on first body use; the remaining flags were section headings, roman
numerals, cited-work names (FAA, FDB, BAF, CORTEX), and reference-list
place abbreviations.

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
