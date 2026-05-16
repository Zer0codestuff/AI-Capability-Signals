# The Frontier Model Oracle

This appendix is the "go wild" part of the project: a serious implementation of a suspicious idea. It asks whether major model launches have release-calendar fingerprints: weekdays, months, moon phases, approximate Mercury retrograde windows and geocentric planetary zodiac signs.

It is not a prediction engine and it is not part of the portfolio headline. It is a demoted exploratory appendix showing how easy it is to manufacture calendar-looking patterns.

## Method

- Input events come from the normalized model table, restricted to GPT, Claude, Gemini, Llama, DeepSeek, Qwen, Mistral and Grok-family releases from 2018 through 2026-05-15.
- Planetary positions use NASA/JPL DE421 ephemeris through Skyfield when available.
- Moon phase is an approximation, marked as approximate.
- Calendar features use a year-preserving random-date baseline.
- Slow-planet zodiac features are not assigned causal p-values because their signs are dominated by the years sampled.

## Pattern Tests

| feature      | top_bucket   |   top_count |   n |    share |   permutation_p_value | null_model                                                                  | approximate_feature   | verdict                      |
|:-------------|:-------------|------------:|----:|---------:|----------------------:|:----------------------------------------------------------------------------|:----------------------|:-----------------------------|
| weekday      | Tuesday      |         151 | 619 | 0.243942 |            0.00019996 | year-preserving random dates                                                | False                 | calendar_cluster_not_causal  |
| month        | April        |          82 | 619 | 0.132472 |            0.00079984 | year-preserving random dates                                                | False                 | calendar_cluster_not_causal  |
| sun_sign     | Pisces       |          69 | 619 | 0.11147  |            0.109289   | uniform bucket permutation                                                  | False                 | no_robust_cluster            |
| quarter      | Q2           |         171 | 619 | 0.276252 |            0.291342   | year-preserving random dates                                                | False                 | no_robust_calendar_cluster   |
| moon_phase   | New          |          90 | 619 | 0.145396 |            0.50035    | uniform bucket permutation                                                  | True                  | no_robust_cluster            |
| mercury_sign | Aries        |          99 | 619 | 0.159935 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| venus_sign   | Pisces       |          94 | 619 | 0.151858 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| mars_sign    | Cancer       |         121 | 619 | 0.195477 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| jupiter_sign | Cancer       |         228 | 619 | 0.368336 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| saturn_sign  | Pisces       |         436 | 619 | 0.704362 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| uranus_sign  | Taurus       |         504 | 619 | 0.814216 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| neptune_sign | Pisces       |         382 | 619 | 0.617124 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |
| pluto_sign   | Aquarius     |         470 | 619 | 0.759289 |                       | not tested; slow planet signs are dominated by the sample year distribution | False                 | temporal_clustering_expected |

## Heatmaps

![Planetary sign occupancy heatmap](../figures/oracle_zodiac_heatmap.png)

![Release weekday family heatmap](../figures/oracle_weekday_family_heatmap.png)

## Interpretation

This is the fun result: release calendars can look patterned even when generated from random dates. The appendix is valuable because it makes that visible. If a striking cluster appears, the next step is not "the planets did it"; it is "check vendor launch schedules, conference calendars, product marketing cycles, earnings windows and data leakage."
