# RR pressure-moments pipeline (SupeRR Selector S2, Round 1)

Pressure = Rajasthan Royals' moments where the champions beat the league and RR fall short, kept only if RR were
below the league in both 2025 and 2026 and RCB above it, and only on live balls (leverage >= median). Each ball counts once.
Players are scored on 2025-26 (IPL 2026 1.0, IPL 2025 0.7, top-nine T20I 2026 0.5, 2025 0.35 effective balls):
SCORE = P (pressure level) + D (pressure minus own usual level), empirical-Bayes shrunk, same code for bat and ball.

Run: `RAW_DIR=/path/to/supplied/data ./run_all.sh`
Steps: s01-s04 data, win probability, leverage, ball values; s00 RR moment masks (rebuilt from the earlier moment
search, scripts/s38); s00b hold-up test + pressure flag; s05 ratings; s06 reliability; s07 sensitivity; s09 lists;
s10 dashboard (out/dashboard.html); s11 PDF report (out/SupeRR_Round1_Report.pdf). All settings: config.py.
2027 squad/auction status (inputs/) is hand-entered from news, not supplied data.
Known dependency: s00 reads /home/claude/scripts/s38_all_moments.py to rebuild the original 48 moment definitions.
