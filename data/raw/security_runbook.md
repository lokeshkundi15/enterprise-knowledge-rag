# Infrastructure Security & Incident Runbook

## SEC-01: Production Credential Rotation Procedure
All production database passwords and API tokens must be rotated every 90 days.
Steps:
1. Generate a new cryptographically secure secret in AWS Secrets Manager.
2. Update the secondary credential slot in the Kubernetes Secret manifest.
3. Trigger a rolling restart of the target microservice deployment using `kubectl rollout restart deployment/<service-name>`.
4. Verify application connectivity via Prometheus metrics endpoint.
5. Invalidate the old secret after a 24-hour grace period.

## SEC-02: SSH Bastion Access Policy
Direct SSH access to production EC2 instances is strictly prohibited. Engineers must connect via the Teleport Bastion gateway using short-lived certificate authentication tied to Okta Single Sign-On (SSO). Emergency break-glass access requires dual-manager approval via PagerDuty incident channel #sec-break-glass.

## SEC-03: Incident Ticket Naming Convention
All infrastructure security incidents must follow the naming standard: `SEC-INC-<YYYY>-<4-digit-ID>` (e.g., `SEC-INC-2026-0842`).