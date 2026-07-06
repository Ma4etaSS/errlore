# errlore error-reduction A/B — model claude-haiku-4-5

tasks (test): 96 | families active (had seed failures): ['csv_order', 'id_norm', 'letter_sent', 'log_ts', 'reverse', 'round_rule']

| arm | failures | fail rate |
|---|---|---|
| A plain | 63/96 | 65.6% |
| B errlore | 20/96 | 20.8% |

discordant pairs: errlore fixed 49, errlore broke 6
exact McNemar p = 1.823e-09
repeat-error reduction: 68.3%

### KNOWLEDGE-GAP (workspace conventions): A 46/48 -> B 0/48 | reduction 100%

### CAPABILITY-GAP (model skill limits): A 17/48 -> B 20/48 | reduction -18%

per-family (fail A -> fail B):
- csv_order: 12 -> 0 *lesson active*
- id_norm: 12 -> 0 *lesson active*
- letter_sent: 6 -> 8 *lesson active*
- log_ts: 12 -> 0 *lesson active*
- mult4: 2 -> 1
- nth_char: 0 -> 0
- reverse: 9 -> 11 *lesson active*
- round_rule: 10 -> 0 *lesson active*

raw outputs: /tmp/errlore_ab_4iljcym1/raw_outputs.jsonl
errlore stats: {'errors_total': 29, 'errors_resolved': 29, 'errors_unresolved': 0, 'lessons_total': 23, 'lessons_applied': 14, 'pending_injections': 0, 'trust': {'claude-haiku-4-5': 0.92}}