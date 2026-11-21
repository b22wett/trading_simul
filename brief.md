# Dataset Interview, rehearsal 2 (instruction booklet)

**Rules.** Work alone, 9:00-15:00. Internet and documentation allowed; no AI tools, no help from anyone. Send me your notebook/script, slides and one-page summary at 15:00. Debrief afterwards.

## Context
The risk team at a trading firm needs a forecast of how volatile each of 25 stocks will be **tomorrow**, so that position limits can be set at the end of each day. They currently use "today's volatility". They ask:

> "Can you do better than our current rule? Build something we can trust, tell us how much better it is, and tell us when we should not rely on it."

## File
`stocks_daily.csv`: one row per stock-day.
- `date`, `ticker`, `sector`
- `open`, `high`, `low`, `close`, `volume`
- `realised_vol`: the day's realised volatility, computed by the vendor from intraday data
- `earnings_day`: 1 if the company reports earnings that day (dates are announced weeks in advance)
- `news_count`: number of news items that day
- `vol_state`: vendor-supplied qualitative label of the stock's volatility state (low / normal / high)

The data comes straight from the vendor. Nobody has checked it.

## Deliverables
1. A reproducible notebook/script.
2. A short deck (5-7 slides) for the risk managers, who are not statisticians.
3. A one-page written summary: data issues and how you handled them, approach, validation, results with honest uncertainty, risks, next steps.

## Timeline (suggested)
- 9:00-10:30 audit and explore (document every decision)
- 10:30-12:30 baseline, features, validated model
- 12:30-13:00 lunch (take it)
- 13:00 freeze modelling; build slides and summary
- 15:00 send

## What the interviewer cares about
Auditing messy data, a clear question and target, avoiding leakage, a simple baseline first, honest uncertainty, and a deck with a clear message and labelled figures.
