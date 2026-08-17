# Information Security & PCI-DSS Compliance

## COMP-01: Data Encryption at Rest and in Transit
Customer Personally Identifiable Information (PII) and Primary Account Numbers (PAN) must be encrypted at rest using AES-256 GCM. All internal microservice-to-microservice communications must mandate TLS 1.3 encryption.

## COMP-02: User Password Complexity Policy
All employee and customer passwords must be a minimum of 14 characters, contain at least one uppercase letter, one lowercase letter, one number, and one special character. Account lockout occurs automatically after 5 consecutive failed attempts.

## COMP-03: Production Deployment Change Freeze
A total production deployment change freeze is enforced annually during Q4 peak traffic from November 15 to January 5. Emergency bug fixes during this freeze window require explicit sign-off from both the VP of Engineering and the Head of Infrastructure.