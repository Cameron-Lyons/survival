# Person-years summary dimensions and margins

The public Rust and Python low-level person-years summary checks each table
extent and its product before accessing cells. When totals are requested,
the expanded shape is checked before allocating any margin component.
Overflowing or unaddressable shapes return input errors instead of panicking
or proceeding with wrapped dimension arithmetic. Component lengths still
must match the table shape.

Zero extents preserve valid empty tables without multiplying unused
dimensions. With `totals=TRUE`, stock survival 3.8-12 accepts one-dimensional
empty tables and the two-dimensional shapes `[0, 2]` and `[2, 0]`; their
requested margins are retained. Existing scalar and higher-dimensional empty-table conventions
also remain supported, including time-cut count margins.

Margin accumulation uses direct column-major slab offsets. Previously,
every input and output cell lookup allocated a multi-index vector and
recomputed strides. The row, column and grand-total addition order remains
unchanged, as do rate, ratio and confidence-interval calculations.

Three Rust public API tests and 86 Python controls cover 54 invalid calls and
32 valid controls. Invalid dimensions are tested without allocating large
input tables; valid controls compare independent hand sums, empty margins,
time-cut conventions, rates, ratios and closed-form confidence limits.

## Complete-call measurements

The benchmark compares preserved release extensions before and after the
dimension guards and direct-offset change, using the same final Python
wrapper. Eight benchmark cases cover two- and three-dimensional tables
with 250,000 and 500,000 cells. Each requests rates and ratios, with confidence
intervals either disabled or enabled for both measures.
All 15 returned fields are checked bit-for-bit outside the timer.

Each trial uses three warmups and nine alternating before/after pairs with
normal garbage collection. The second trial reverses case and initial
execution order. Input preparation and output verification are outside the
timer; native input conversion, margin construction, calculations and public
result construction are included. Both releases use identical ordinary
tables; invalid-dimension cases are correctness checks rather than timings.
