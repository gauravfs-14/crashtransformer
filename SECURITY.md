# 🔒 Security & Secrets

Follow these guidelines to keep credentials and data safe.

## Secrets handling

- Store API keys only in `.env` (or a secret manager), not in CLI or code
- Never commit `.env`; share `.env.example` instead
- Use hidden input in the interactive setup; keys are not echoed
- Rotate keys periodically and revoke unused ones

## Least privilege

- Use provider keys scoped to the minimum permissions required
- Use separate keys per environment (dev/staging/prod)

## Logs and data

- Avoid logging sensitive values; the logger avoids printing keys
- Redact narratives if they contain PII before processing
- Control access to `logs/` and `artifacts/` directories

## Network & Neo4j

- Use strong passwords for Neo4j; avoid default credentials
- Restrict Bolt access by IP; use SSL/TLS where available
- Place DB close to compute to limit exposure and latency

## Reporting vulnerabilities

Open an issue with a non-sensitive description or contact the maintainers privately. Do not include keys, credentials, or PII in reports.
