1. Audit. Remove duplicates. Check describe(), info(). Check dates (Weekdays? Gaps?) Check changes and level -- large change may indicate collection shift. Conceptually -- consider universe, past and future for each column. Put in assertions.
2. Features. Returns/changes over levels.
3. Baseline. dummy_classifier or similar.
4. Model. Keep it simple, use LinearRegression o.s. GLM for special distribs. Use log-loss as metric for logistic regression. HGBR if time allows, no tuning hypers.
5. Uncertainty. Consider conditions under which the model no longer works. Consider biases of the data and uncertainties.

GIVE AN HOUR AT LEAST FOR PRESENTATION PREP.
