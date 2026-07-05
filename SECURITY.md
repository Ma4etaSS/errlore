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
Email postoluk.m2@gmail.com. No bounty program; fixes are prioritized and
credited in the changelog.
