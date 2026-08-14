# Reference list verification — 13 Aug 2026

Every entry in `build_manuscript.py` checked against the publisher record.
Auditable in the same spirit as `PROVENANCE.md`: what was checked, what was
wrong, what is still open.

**Method.** Web search against publisher pages (Nature, Cell, PNAS, eLife,
J Neurosci, Annual Reviews, IEEE, bioRxiv) plus an automated cited-vs-listed
audit over the manuscript source. Not verified by opening every PDF.

---

## Errors found and fixed

| # | Entry | Problem | Fixed to |
|---|---|---|---|
| 1 | **Pennock et al. 2023** | **Author omitted — Yihan Wu missing** from the list | Pennock IML, Racey C, Allen EJ, **Wu Y**, Naselaris T, Kay KN, Franklin A, Bosten JM |
| 2 | **Akbarinia 2025** | **Listed twice**, with conflicting titles: "The *footprint* of colour in EEG signal" and "The *hallmark* of colour in EEG signal" | Duplicate deleted; "**hallmark**" is correct (confirmed on bioRxiv) |
| 3 | **Prince et al. 2022 (GLMsingle)** | **Cited in Methods, absent from the reference list** | Added: eLife 11:e77599 |
| 4 | **Lin et al. 2014 (COCO)** | **Cited in Methods, absent from the reference list** | Added: ECCV 2014, pp 740–755 |
| 5 | **Bartsch et al. 2026** | Discussion made an EEG/MEG claim (pre-stimulus colour readout) **with no citation at all** — it sat in a `[Add: …]` placeholder | Preprint located and cited: bioRxiv 2026.06.11.731729 |
| 6 | **Akbarinia 2025** (second instance) | The "colour improves object decoding" clause in the Discussion was **also uncited** | Citation added |
| 7 | **Allen et al. 2022** | `et al.` in a 13-author reference | Full author list |
| 8 | **Doerig et al. 2025** | `et al.` mid-list; volume and pages missing | Full list; Nature Machine Intelligence 7(8):1220–1234 |
| 9 | **Kim et al. 2020** | `et al.`; pages missing | Kim I, Hong SW, Shevell SK, Shim WM; PNAS 117(23):13145–13150 |
| 10 | **Scotti et al. 2024** | Placeholder — no venue, no pages | Full 11-author list; ICML, PMLR 235:44038–44059 |
| 11 | **Bartels & Zeki 2000** | Subtitle dropped | "…: new results and a review" |
| 12 | **Pennock et al. 2026** | Issue number missing | PNAS 123(15):e2535986123 |
| 13 | Reference list | **Not alphabetical** — Doerig after Hansen, Akbarinia after Kim, Lafer-Sousa after Pennock | Sorted; `van de Weijer` filed under V per J Neurosci convention |

Items 1, 3, 4, 5 and 6 are the substantive ones. A missing co-author and two
uncited claims are the kind of thing a reviewer notices.

---

## Verified correct as written — no change needed

Bannert & Bartels 2013 (*Curr Biol* 23(22):2268–2272) · Brainard & Freeman 1997
(*JOSA A* 14(7):1393–1411) · Brouwer & Heeger 2009 (*J Neurosci*
29(44):13992–14003) · Hansen et al. 2006 (*Nat Neurosci* 9(11):1367–1368) ·
Lafer-Sousa et al. 2016 (*J Neurosci* 36(5):1682–1697) · Naselaris et al. 2011
(*NeuroImage* 56(2):400–410) · Teichmann et al. 2019 (*NeuroImage* 200:373–381) ·
van de Weijer et al. 2009 (*IEEE TIP* 18(7):1512–1523) · Witzel & Gegenfurtner
2018 (*Annu Rev Vis Sci* 4:475–499) · Zeki 1973 (*Brain Res* 53(2):422–427) ·
Zeki 1980 (*Nature* 284:412–418) · Zeki 1990 (*Brain* 113(6):1721–1777)

**Note on the Allen 2022 title.** The manuscript has it right — the published
*Nature Neuroscience* title is "…to bridge cognitive neuroscience and artificial
intelligence". `READING_LIST.md` had the **bioRxiv preprint** title ("…cognitive
and computational neuroscience"); that has been corrected.

---

## Closed by Nina — the last open entry

**Goddard E, Marchenko K, Clifford CWG (2025)** fMRI responses to color and
objects in the ventral visual pathway. *Journal of Vision* **25(9):2913**.
doi:10.1167/jov.25.9.2913. [VSS meeting abstract]

I could not confirm volume/issue automatically — ARVO returns an empty response
to automated fetching and the Chrome extension was unavailable. Nina supplied
the DOI from the article page on 13 Aug; `10.1167/jov.25.9.2913` encodes
volume 25, issue 9, article 2913, which matches the September-issue inference.
**No `[confirm]` markers now remain in the reference list.**

---

## New reference added

**Bartsch MV, Spaak E, de Lange FP (2026).** Colourful predictive templates in
early visual cortex. *bioRxiv* 2026.06.11.731729.

Donders Institute, Radboud. EEG+MEG, n=37, coloured discs in a predictable
sequence; a decoding model trained on a separate colour localiser reconstructs
the *predicted* colour during the pre-stimulus period, while only a grey
placeholder is on screen.

This is a good citation for you beyond filling the gap — it is direct evidence
that early visual cortex carries anticipatory colour content, which is the
temporal half of the predictive-coding reading your Discussion advances and
which fMRI cannot supply. Worth reading properly; added to `READING_LIST.md`.

---

---

## A tension the new citation exposed — and the fix

Adding Bartsch et al. created an internal contradiction that the vaguer wording
had hidden. The Discussion cited a paper showing **predicted** colour in **early**
visual cortex, then two clauses later described early cortex as carrying the
"bottom-up, immediate" signal. Adjacent sentences pulling opposite ways.

**Nina caught this**, and it was worth catching. Two changes:

1. **The region parentheticals are gone.** "(early, immediate)" and "(higher
   visual cortex, delayed and fed back)" were removed, so the contrast is now
   between *signal types* — bottom-up sensory versus top-down object-predicted —
   without claiming that either is confined to one region. This is more accurate
   independent of Bartsch: the manuscript's own Introduction cites Bannert &
   Bartels (2013), where greyscale bananas evoke yellow in **V1**. Top-down
   colour has reached early cortex in the literature for over a decade.
2. **Limitations now concedes the residual may not be purely sensory:** "…early
   cortex can itself host predicted colour (Bartsch et al., 2026), so the
   residual need not be purely sensory." This pre-empts a reviewer question the
   data genuinely cannot settle.

**Why none of this touches the result.** The residualisation removes variance
predictable from **80 COCO object categories**. Bartsch's paradigm contains no
objects at all — coloured discs in a predictable temporal sequence — so its
prediction is sequence-driven, not category-driven, and category regressors
would not capture it. The two measure different things.

If anything the finding reads better afterwards: given that early cortex *can*
host predicted colour, showing that the early signal is nonetheless the part
**least** explained by object identity is informative rather than trivial. It
argues the early residual is not merely leaked semantic inference.

---

## Post-verification state

| Check | Result |
|---|---|
| Citations in text missing from reference list | none |
| Reference list alphabetical | yes |
| Duplicate entries | none |
| Entries still marked `[confirm]` | **none** |
| Uncited claims in the Discussion | none remaining |
| Abstract / Significance / Introduction / Discussion | 247 / 109 / 648 / **1493** — all within limits |
| Document rebuilds | yes |

Re-run the audit any time with the snippet in `CHANGES_PENDING.md` Part 5, or
`python count_words.py <build_manuscript.py>` for the counts alone.
