# Second fixture search, before any labels or fitting

The first declared analytic search stopped without a qualifying fixture.
Its complete negative result is preserved in fixture_recon.json. Width 20
had completion 0.999505320284508; widths 21 onward were unreachable at T=24.
No numerical-domain error occurred. No demonstrations have been sampled and
no parameters have been fitted. The first search is not retroactively changed.

Declare a second analytic-only search. Use a 3xW straight corridor, all cells
in rows 0 and 2 occupied, row 1 free. Start=(1,0,E), goal=(1,W-1), four
headings, standard d_safe=2 features, T=24, truth=[1,0.7,1.2], M=30.
Search W=5,...,18 in order and take the first exact completion in [0.70,0.90].
If this family also fails, stop for user review without labels or fitting.
Numerical limits remain unchanged and fatal. Save full covariance and its
rank/condition diagnostics, even if this corridor introduces weakly identified
directions. Everything else in protocol.md remains frozen.
