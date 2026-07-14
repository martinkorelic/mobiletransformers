# Changelog

All notable changes to MobileTransformers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow Semantic Versioning
once v1.0 is released.

> Skeleton created by #31; the release-history entries and versioning policy are finalized by #32
> (`05_code_plans/05`).

## [Unreleased]

### Added
- One-command export CLI + Hub package format (manifest-first cache bridge).
- Hub pull/install + adapter push-back (Python).
- `VectorStore` boundary (`InMemoryVectorStore` for JVM tests).
- Developer `Makefile` + staged CI (`ci.yml`) + device pipeline (`device.yml`).
- Docs set (partial): `EXPORT.md`, `RAG.md`, `PUBLIC_API.md`, generated `COMPATIBILITY_MATRIX.md`.

### Non-goals (this line)
- On-device engine/facade parity (device-gated): tracked separately until device testing resumes.
