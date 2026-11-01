# Phase 4b — writer self-scoring (ARS academic-paper full, contract academic_paper/writer_full/v1)

Draft: `docs/paper_b/draft/main.tex` → `main.pdf` (main text + Generative AI Use on pages 1–10; references 11–12). Long formats kept at the user's request: `long/v1_full_length.pdf` (16 pp, ≈ 4 700 words) and `long/v2_fourteen_pages.pdf` (14 pp, ≈ 4 300 words).

## Draft Body
In `main.tex` (not repeated here).

## Dimension Scores
| Dim | Name | Score | Evidence |
|---|---|---|---|
| D1 | section_completeness | pass | Abstract, keywords, §1–§8, Reproducibility, Generative AI Use all written. Placeholders: `[to confirm]` (corresponding author's email, user-owned) and `[repository URL]` (deferred to the Nov 10 go/no-go with the password decision). Table 2 moved to the supplement (`supplement_table_b1.tex`), announced in the text. |
| D2 | citation_density | pass | 21 of the 23 verified entries cited; every prior-work claim carries one of them; every empirical number traces to a named round whose record is in the evidence map (D159/D160 for B1–B4). No citation outside the verified list. Two verified entries (Nakamura 2025, Lee 2025) were dropped from the related-work paragraph for space; they remain in `references.bib` uncited. |
| D3 | argument_blueprint_fidelity | pass | Section order and claims follow the skeleton; every FAIL verdict is written as the registered consequence sentence (H-B1-3 scoped to 98/2 %; H-B1-4 dwell law dropped, R3 demoted; H-B2-2 "typically" with 97.8 %; H-B2-4 "cannot leave the trap"; H-B3-1 "present but weaker"). User decisions 2–7 applied verbatim. |
| D4 | total_word_count | warn | ≈ 3 400 words of body text against the skeleton's 5 310. The skeleton's budget assumed ~530 words per PMLR page with 8 figures; the class holds ~500 words per page and the figures take ~2.3 pages, so 10 pages leave ≈ 3 400 words. The budget, not the draft, was wrong; the user has the long formats. |
| D5 | per_section_word_count | warn | Same cause; every section is below its allocation by a similar proportion. |
| D6 | paragraph_structure | pass | Topic sentence → number with N → reading → link, throughout §5–§6; transitions are sentences. |
| D7 | register_consistency | pass | Formal register; 0 em dashes; no flagged filler terms; "pinned" absent from text and captions. Semicolons are frequent (≈ 90 in the body) because number lists were compressed for the page limit; flagged for the evaluator. |

## Failure Condition Checks
- F1 (mandatory block): none.
- F4 (mandatory warn): none.
- F2 (high-priority block): none.
- F3 (high-priority warn): D4, D5 (word count below target because of the page limit; justified above).
- F0: all mandatory dimensions pass.

## Writer Decision
`writer_decision=accept` with the D4/D5 warns recorded: the word budget in the Paper Configuration Record should be revised to ≈ 3 400 words for a 10-page PMLR paper with eight figures. Items for the evaluator / panel: figure legibility at the reduced widths (Figs. 1, 6, 7 are small), semicolon density, and whether Table 2's move to the supplement weakens §5.1.
