# Placeholder index

Everything in the manuscript that renders **red** in the compiled PDF, and what
closes it. Nothing here can survive into a submitted version.

Macros: `\ph{NAME}` = a named gap · `\phd` = a placeholder table cell ·
`\phnote{...}` = an inline editorial note.

## Status: all clear (2026-09-04)

```bash
grep -n '\\ph{\|\\phd\|\\phnote\|\\tba' paper/main.tex   # only the definitions
```

Every placeholder is closed. The macros are left defined so that a new gap
introduced later still renders red rather than hiding; **none is in use**.

## How each was closed

| Location | Placeholder | Closed by |
|---|---|---|
| Abstract, §1, §6.2, Table 1 | graph-ablation result | R5, R6 and R11 landed; Table 1 carries all three plus the trivial row; Table 3 records the verdict on each |
| Title block | author list, affiliations, contact | `Anonymous ACL submission` + a statement that names, affiliations, funding and CRediT are withheld under ARR anonymous review |
| Figure 1 | system diagram | `paper/figures/pipeline.tex`, pure TikZ, vector; Algorithm 1 now states the same loop formally |
| §2 | MoniTor / ASK-HINT positioning | MoniTor cited as the anomaly-vs-procedural contrast; ASK-HINT removed (no bibliographic record exists) |
| §2 | Assembly101 / HoloAssist / EPIC-Tent / Ego4D bib | written and cited |
| §2 ×3 | Seminara, Spurio, reading depth | §2 rewritten; every citation verified against arXiv or proceedings |
| §5.3 | offline protocol | dropped, with the reason stated in the text rather than the omission left silent |
| §6.5 | qualitative figure | Figure 7, built from the score stream and typed decisions (the free text is degenerate, so none is quoted) |
| §Limitations | second dataset | scoped to cooking, with the binding constraint stated: no other procedural-error benchmark ships gold task graphs |
| §Limitations | training seeds | stated as a limitation rather than run; justification is the interval width |
| §Ethics | dataset licence | Apache 2.0, stated |
| §AI disclosure | venue policy | written; ACL/EMNLP place it in Acknowledgments at camera-ready, noted there |
| §Reproducibility | code URL | `\coderepo` macro, set at submission |
| §Acknowledgments | funding | `\fundingnote` macro, set at camera-ready |

## The two values the authors still set

Both are macros at the top of `main.tex`, deliberately not invented:

```latex
\newcommand{\coderepo}{...}    % anonymised artefact URL, at submission
\newcommand{\fundingnote}{...} % funding + thanks, at camera-ready
```

## Before submitting

```bash
python paper/gen_tables.py && (cd paper && latexmk -pdf main.tex)
grep -c Overfull paper/main.log      # must be 0
grep -n '\\ph{\|\\phd\|\\phnote' paper/main.tex | grep -v newcommand | grep -v '^3[6-9]:\|^4[0-9]:'
```

Swap the `documentclass` block for the official `acl.sty` (see the header
comment in `main.tex`) once the venue is fixed; `acl.sty` is not installed on
this host, so the draft compiles with the article class.
