# Entry replay over the scanned universe, 2026-08-31

Question: is there an entry rule or a filter, on the data this engine already
collects, with positive expectancy under the shipped exits (take +20% or leave
at five minutes)?

Method. A deterministic 7.7% sample (`at % 13 = 0`) of first-sight decisions
over the last 47 hours: 5,073 tokens, 51k minute ticks (listing plus follow-up),
creator facts computed the way the tick computes them (prior labeled launches
before this one, capped at 50; prior deaths, capped at 5). Entry at a tick's
mark times 1.018 (measured fill), exit at a tick's mark times 0.969 (measured
haircut). Tokens with no priceable tick after entry are reported both excluded
and as -100% (`cons`), which is how the engine books an unroutable exit.

Entry rules: E0 first sight; E1 the shipped confirmation (third observation,
price above first sight, liquidity holding); E2 pullback (10% off the running
high with liquidity above first sight); E3 activity (first tick with $800 in
the curve, the pre-08-31 floor).

## Result (net %, TP20/hold5)

| entry | n | mean | ex-best | median | win | cons | by day |
|---|---|---|---|---|---|---|---|
| E0 first sight | 4,770 | -5.1 | -5.6 | -4.9 | 5% | -10.8 | -3.6 / -6.1 / -6.5 |
| E1 confirm | 1,798 | -5.0 | -5.3 | -5.0 | 4% | -12.2 | -4.9 / -5.2 / -4.8 |
| E2 pullback | 71 | -3.1 | -12.2 | -20.0 | 32% | -4.4 | -6.8 / +4.2 / -35.2 |
| E3 activity | 174 | -6.9 | -8.9 | -5.3 | 44% | -6.9 | -7.5 / -4.2 / -17.7 |

The median of E0 and E1 is the round-trip cost to one decimal: at one-minute
resolution 95% of launches do nothing inside five minutes. The 5% that move
average negative. Holding ten minutes is worse (E0 -7.5%). Every take-profit
level from 15 to 30 lands within 0.2 points of the others.

Stratified (E0, TP20/hold5), the only cells with a positive mean: "first
launch and $300+ at entry" +1.4% (n=65, ex-best -3.9), "prior 4-10 launches"
-0.8% (n=472, ex-best -5.1, the whole mean is one token). Creator history
separates death rate (documented in the policy) but not five-minute return:
rugs=0 -5.2% against rugs>0 -5.3%. First-sight liquidity: under $300 -5.1%,
$300-800 -9.1%, $800+ -5.2% with a 46% win rate and a fatter loss tail. Hour of
day: every hour between -9% and +1%, the two positive hours are one point.

## Reading

There is no pocket here. The signal the corpus does carry, creator history,
predicts whether a token dies within 30 minutes, not whether it moves in the
first five, and this engine trades the first five. What moves these tokens
happens in their first seconds, before this pipeline's first sight, which is
a cron minute plus a listing poll away. That is an architecture statement,
not a parameter.

What this leaves: the paper engine runs at near-zero cost as a measurement
instrument under one fixed policy, reports daily, and the record under hash
`690a62cf` is the first honest one. Any claim of edge from here has to come
from a different vantage point on the same market (sub-second ingestion at
creation, or a corpus feature no one else has), or from a different market.
It will not come from another sweep of these dials.
