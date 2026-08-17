# Engineering RFC & Architecture Standards

## RFC-101: Microservice Database Connection Pooling

All Java/Spring microservices must configure HikariCP connection pools. The maximum pool size should not exceed 20 connections per pod. Minimum idle connections must be set to 5, and connection timeout must be configured strictly to 30000ms (30 seconds) to avoid thread exhaustion.

## RFC-102: API Idempotency Standard

All non-GET REST endpoints that perform mutations must implement an `Idempotency-Key` HTTP header. The service must store this key in Redis with a Time-To-Live (TTL) of 24 hours. If a duplicate key is received, the cached response payload must be returned directly with HTTP status 200 OK.

## RFC-103: Kafka Message Retention Policy

Topic partitions in production Kafka clusters have a default retention period of 7 days (168 hours). For compliance audit topics, retention must be configured to 365 days with log compaction enabled.
