# Security policy

## Supported versions

Until the first stable release, security fixes are provided on the latest beta
line only. After 1.0, this document will list supported stable release lines.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository:

https://github.com/Wouittone/differential-equations-rs/security/advisories/new

Do not open a public issue for an undisclosed vulnerability. Include affected
versions, reproduction details, impact, and any known mitigation. Maintainers
will acknowledge the report, assess severity and affected releases, coordinate
a fix and disclosure, and credit reporters who wish to be named.

Dependency and workflow risks are governed by `SUPPLY_CHAIN.md` and enforced in
CI with the committed lockfile and `cargo-deny` policy.
