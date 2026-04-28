# ru-jailbreak-guard

Russian-language jailbreak detection service. End-to-end MLOps pipeline as
the final project for the OTUS MLOps course.

## Цель

Бинарный классификатор текстов на русском языке:
содержит ли входное сообщение попытку jailbreak
(обхода safety guidelines LLM). Сервис задумывается
как составная часть guardrails-слоя для
русскоязычных LLM-приложений.

Курсовой проект сфокусирован на
**процессах и автоматизации**, а не на качестве
модели. Модель — простая бинарная классификация
на эмбеддингах, нужна для иллюстрации полного
MLOps-цикла.

## Стек

| Слой | Технологии |
|------|------------|
| Облако / IaC | Yandex Cloud + Terraform |
| Хранилище данных | S3 (Yandex Object Storage) + DVC |
| Препроцессинг | Polars / Pandas (текстовые данные) |
| Эмбеддинги / модель | ruBERT-tiny2 + scikit-learn / LightGBM |
| Эксперименты / Registry | MLflow (Postgres backend) |
| Оркестрация | Flyte |
| Сервинг | KServe (RawDeployment) + Streamlit UI |
| Контейнеризация | Docker, GitHub Container Registry |
| Деплой | Kubernetes (managed) + ArgoCD GitOps |
| CI/CD | GitHub Actions |
| Мониторинг | Prometheus + Grafana + Alertmanager |
| Бенчмарк | HiveTrace GLiNER (offline reference) |

Источники данных — `dmtrdr/russian_prompt_injections`
(HuggingFace, 22K примеров) + HiveTraceRed (русский subset)
+ WildGuardMix (cross-validation)
+ `hivetracered`-генерация adversarial-вариаций.

## Релизы

- ✅ `v0.1.0` — Phase 0: bootstrap (Python skeleton, CI, ArgoCD locally)
- ✅ `v0.2.0` — Phase 1: data pipeline (DVC + HiveTrace sources)
- ✅ `v0.3.0` / `v0.3.1` — Phase 2 + 2.5: TF-IDF + LightGBM models live (KServe)
- ✅ `v0.4.0` — Phase 3: fine-tuned ruBERT-tiny2 model live
- ✅ `v0.5.0` — Phase 4: Streamlit multi-model UI
- ✅ `v0.6.0` — Phase 5: Flyte orchestration + cron retraining
- ✅ `v0.7.0` — Phase 6: Prometheus + Grafana + alerting
- `v0.8.0` — Phase 7: closed-loop retraining (drift trigger + automated promotion)
- `v1.0.0` — Phase 8: cloud demo on YC managed k8s

## Установка локально

См. [docs/local-setup.md](docs/local-setup.md). Кратко:

```bash
git clone https://github.com/Meffazm/ru-jailbreak-guard.git
cd ru-jailbreak-guard
make setup            # uv sync
make all              # lint + type-check + tests
```

## Структура

```
ru-jailbreak-guard/
├── src/ru_jailbreak_guard/   # Python package (importable)
├── tests/                    # pytest, mirrors src/
├── gitops/                   # ArgoCD watches this directory
│   ├── applicationset.yaml
│   ├── project.yaml
│   ├── apps/                 # populated in Phase 2+
│   └── envs/{local,cloud}/   # environment-specific Helm values
├── docs/
│   ├── adr/                  # architecture decision records
│   └── local-setup.md
├── .github/workflows/        # CI
├── pyproject.toml            # uv-managed
├── ruff.toml
└── Makefile
```

## Phase 2 — Cheap models served via KServe

- TF-IDF + Logistic Regression classifier
- LightGBM on ruBERT-tiny2 mean-pooled embeddings
- MLflow tracking + registry (Postgres + MinIO via Helm/ArgoCD)
- Two KServe RawDeployment InferenceServices, deployed via the existing ArgoCD ApplicationSet
- Custom FastAPI predictors with a locked response schema across families
- Multi-arch image builds (amd64 + arm64) pushed to GHCR

См. [docs/runbooks/phase-2-end-to-end-smoke.md](docs/runbooks/phase-2-end-to-end-smoke.md) для пошагового локального прогона.

## Документация

- [docs/local-setup.md](docs/local-setup.md) —
  установка локального окружения
- [docs/runbooks/](docs/runbooks/) — operational runbooks
- [docs/adr/](docs/adr/) — архитектурные решения

## Лицензия

MIT — см. [LICENSE](LICENSE).
