# ru-jailbreak-guard

Russian-language jailbreak detection service. Bachelor of MLOps final project (otus.ru).

Бинарный классификатор для русскоязычных промптов: содержит ли вход попытку
jailbreak (обход safety guidelines LLM). Сервис задумывается как часть guardrails-
слоя для русскоязычных LLM-приложений.

Курсовая работа сфокусирована на **процессах и автоматизации**, а не на качестве
модели — модель умышленно простая, чтобы концентрация была на полном MLOps-цикле.

## Стек

| Слой | Технологии |
|------|------------|
| Облако / IaC | Yandex Cloud + Terraform |
| Хранилище данных | S3 (Yandex Object Storage) + DVC |
| Препроцессинг | Polars (текстовые данные) |
| Эмбеддинги / модели | ruBERT-tiny2 + scikit-learn / LightGBM |
| Эксперименты / Registry | MLflow (Postgres backend) |
| Оркестрация | Flyte |
| Сервинг | KServe (RawDeployment) + Streamlit UI |
| Контейнеризация | Docker, GitHub Container Registry |
| Деплой | Kubernetes (managed) + ArgoCD GitOps |
| CI/CD | GitHub Actions |
| Мониторинг | Prometheus + Grafana + Alertmanager + Pushgateway |

Источники данных: `dmtrdr/russian_prompt_injections` (HuggingFace) +
HiveTraceRed (русский subset) + WildGuardMix (cross-validation) +
`hivetracered`-генерация adversarial-вариаций.

## Быстрый старт

```bash
git clone https://github.com/Meffazm/ru-jailbreak-guard.git
cd ru-jailbreak-guard
make setup            # uv sync
make all              # lint + type-check + tests
```

## Структура

```
ru-jailbreak-guard/
├── src/ru_jailbreak_guard/   # Python package
├── tests/                    # pytest, mirrors src/
├── flyte/workflows/          # Flyte tasks, workflows, launchplans
├── gitops/                   # ArgoCD watches this directory
│   ├── apps/                 # k8s manifests (predictors, monitoring, …)
│   └── helm-values/          # Helm value inputs (rendered into apps/)
├── infra/                    # Terraform — Yandex Cloud
├── scripts/
├── docker/                   # Dockerfiles for predictors, trainer, UI
├── .github/workflows/        # CI (tests, image builds, auto-promote)
├── pyproject.toml
└── Makefile
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
