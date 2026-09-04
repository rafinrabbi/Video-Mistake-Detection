# Citation audit

Every entry in `refs.bib` and its verification status. Rule for this project:
**no citation enters the manuscript until its bibliographic record has been
checked against arXiv or the official proceedings page.**

## Verified — record confirmed (31 Aug 2026)

| Key | Check |
|---|---|
| `peddi2024captaincook4d` | arXiv 2312.14556 + NeurIPS 2024 D&B proceedings page; full 13-author list confirmed |
| `patsch2025mistsense` | ICCV 2025 CVF open-access PDF; 5 authors confirmed |
| `flaborea2024prego` | CVPR 2024 open access + arXiv 2404.01933; pages 18483–18492 |
| `lee2024egoper` | CVPR 2024 open access; pages 18655–18666; 5 authors confirmed |
| `seminara2024taskgraph` | NeurIPS 2024 proceedings (spotlight) + arXiv 2406.01486 |
| `storks2025transparent` | EMNLP 2025 + arXiv 2412.11927; 6 authors confirmed |
| `spurio2026posttraining` | arXiv 2608.28406, accepted BMVC 2026; 6 authors confirmed |
| `lee2026estanet` | arXiv 2606.25317 (24 Jun 2026); 5 authors confirmed |
| `haneji2025egooops` | ICCVW 2025 (SAUAFG) + arXiv 2410.05343 |
| `bai2025qwen25vl` | arXiv 2502.13923 |
| `hu2022lora` | ICLR 2022 + arXiv 2106.09685 |
| `plini2024tiprego` | arXiv 2411.02570 — key renamed from `flaborea2024tiprego`; first author is Plini. Done. |
| `monitor2025` | arXiv 2510.21449, NeurIPS 2025; 5 authors confirmed |
| `sener2022assembly101` / `wang2023holoassist` / `jang2019epictent` / `grauman2022ego4d` | benchmark citations, records confirmed |

## UNVERIFIED — must be read or removed

Asserted in the project's planning documents but **not** confirmed and **not**
currently cited in `main.tex`:

- **MoniTor** — RESOLVED: arXiv 2510.21449 (NeurIPS 2025), full author list in refs.bib. It is surveillance anomaly detection, not procedural; cited as the contrast. Originally described in our notes as a training-free online video anomaly
  detector. No bibliographic record located. Do not cite until found.
- **ASK-HINT** — described as LLM-prompted procedural guidance. No record located.
- **Assembly101 / HoloAssist / EPIC-Tent / Ego4D** — RESOLVED: bib entries
  written and cited in the benchmark paragraph of §2.

## Cited but not read at comparison depth

These have confirmed records and are cited, but the manuscript's characterisation
of them is drawn from abstracts and paper pages, not close reading. The `\phnote`s
they carried in §2 are gone and the characterisations were rewritten from the
papers; the depth caveat below is retained as a record of how far the reading
went:

- `seminara2024taskgraph` — **highest priority**: same dataset, same task, task
  graphs. A reviewer will expect a direct comparison.
- `spurio2026posttraining` — contemporaneous; decide concurrent-work framing.
- `lee2026estanet`, `flaborea2024tiprego` — cited for positioning only.
