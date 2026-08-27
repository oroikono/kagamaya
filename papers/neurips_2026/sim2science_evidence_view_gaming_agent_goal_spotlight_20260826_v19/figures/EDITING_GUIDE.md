# Editing the figures manually

## Figure 1: selected evidence view versus cross-view admission

The source is `figure1_editable.tex`; it is native TikZ and remains fully
editable in Overleaf. Compile `figure1_standalone.tex` for a fast preview, then
compile the full paper because figure height affects page flow.

The figure has two semantic panels:

- **Panel A** presents the controlled contrast in plain language: the selector
  chooses one of two honest temporal views, both enter the same fixed
  evaluator, and they yield different scientific conclusions.
- **Panel B** is a separate choice-independent comparison rule. It may retain
  a common conclusion or withhold the claim and request more evidence.

Safe edits:

- Panel A spans the full width at `y=1.70--4.47`; Panel B is the full-width
  strip at `y=0.04--1.64`.
- the selector has a dedicated card centered at `x=1.04`; preserve at least
  1.5 mm between text and its border;
- coarse/denser view cards are centered at `x=4.05`, at `y=3.42/2.30`;
- the frozen-verifier card is centered at `x=7.20`;
- terminal cards are centered at `x=11.28`, with deliberately smaller type and
  at least 5 pt horizontal inner padding;
- the required-view group, guard, and two guarded outcomes are centered at
  `x=2.53`, `6.38`, and `10.93`, respectively;
- arrows use named node anchors and protected gutters. Do not replace their
  anchors with literal endpoints at box borders;
- `cbblue`, `cborange`, `cbvermillion`, and `cbcharcoal` define the
  colorblind-safe palette.

Scientific constraints:

- keep the underlying score-selected view at 7 temporal points, 5 tests, and
  score 13.77;
- keep the denser comparison at 25 temporal points, 2 tests, and score 10.87;
- keep the reader-facing labels `coarse`, `highest score`, `denser`, and
  `recovers terms`; the action IDs are intentionally introduced later in the
  experimental setup, not in this Introduction figure;
- do not put machine terminal names, protocol IDs, vendor/model names, formal
  notation, or held-out support labels back into Figure 1;
- do not merge the per-view evaluator with the cross-view comparison rule;
- do not label agreement as truth, safety, or certification;
- keep the guard demotion-only;
- preserve textual and line-style redundancy if colors change.

The robot and lock are vendored Tabler icon PDFs. Their license and source
record remain under `icons/tabler/`.

## Main results figure: benchmark and agent bars

Open `figure2_editable.tex`. Change only the percentages and labels in the top
edit block. The y positions are calculated from those values automatically.
Values are percentages, not fractions.

The values currently encode:

- false promotions relative to held-out truth: 83.33, 2.28, 0.167;
- selection of the separately flagged coarse view under the accuracy and
  omission goals: 2/188 versus 175/188, 9/158 versus 154/158, and 19/163
  versus 156/163.

If experimental denominators change, edit the values and footnote together.
Do not change a plotted number without updating the corresponding table and
prose in `main.tex`.

## Other editable plots

- `figure2_convergence_editable.tex` contains the temporal-refinement rejection
  count. Residual values are intentionally reported in prose to avoid a dual
  axis or unreadably compressed stacked panels.
- `figure3_mitigation_utility_editable.tex` retains the complete risk/coverage
  comparison moved from the main results figure.
- `figure4_gates_editable.tex` is the appendix construction pipeline.
- `figure5_agent_goals_editable.tex` is the full four-goal agent comparison;
  its line styles and markers must remain redundant with color.

## Visual safeguards

- Keep color redundant with text, border shape, or line style.
- Do not use green as a synonym for scientific truth.
- Keep Figure 1 conceptual; use exact machine statuses only after their Method
  definitions elsewhere in the paper.
- Keep labels inside their cards; do not put prose in an arrow lane.
- Check the figure at ordinary PDF zoom, not only in a standalone crop.
- Also inspect a 300-dpi standalone render for border-touching or clipped text.
- Re-render all five main pages after any geometry change.
