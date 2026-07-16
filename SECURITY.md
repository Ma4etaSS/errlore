# Security Policy

## Supported versions
Latest 0.x release.

## Model
errlore is a local, embedded library: no network calls, no telemetry, no
server. All data lives in plain files under the data_dir you choose —
protect that directory with normal filesystem permissions. Lesson text is
injected into prompts verbatim after sanitization; treat lessons from
untrusted sources as untrusted prompt content.

## Reporting
Email postoluk.m2@gmail.com with a description, affected version, and (if
possible) a minimal reproduction. Please use coordinated disclosure: report
privately first and allow a fix to ship before publishing details.

What to expect (solo-maintainer project, targets not guarantees):
- **Acknowledgment within 72 hours.**
- **Initial assessment within 7 days** — severity triage and whether a fix
  is warranted.
- **Fix for confirmed vulnerabilities within 30 days**, released as a patch
  version with the advisory credited in the changelog (unless you prefer
  anonymity).

No bounty program. If you receive no reply within 7 days, open a GitHub
issue saying you sent a security report (without details) so it can't get
lost in spam.
