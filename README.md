# ru-jailbreak-guard

Russian-language jailbreak detection service. End-to-end MLOps pipeline as the final project for the OTUS MLOps course.

## Цель

Бинарный классификатор текстов на русском языке: содержит ли входное сообщение попытку jailbreak (обхода safety guidelines LLM). Сервис задумывается как составная часть guardrails-слоя для русскоязычных LLM-приложений.

Курсовой проект сфокусирован на **процессах и автоматизации**, а не на качестве модели. Модель — простая бинарная классификация на эмбеддингах, нужна для иллюстрации полного MLOps-цикла.

## Стек

| Слой | Технологии |
|------|------------|
| Облако / IaC | Yandex Cloud + Terraform |
| Хранилище данных | S3 (Yandex Object Storage) |
| Препроцессинг | PySpark на DataProc |
| Эмбеддинги / модель | ruBERT / RuRoBERTa + scikit-learn |
| Эксперименты / Registry | MLflow |
| Оркестрация | Apache Airflow (managed) |
| Сервинг | FastAPI + UI (Streamlit) |
| Контейнеризация | Docker, Yandex Container Registry |
| Деплой | Kubernetes (managed) + HPA |
| CI/CD | GitHub Actions |
| Мониторинг | Prometheus + Grafana + Alertmanager |

## Статус

В стадии планирования. Архитектура, выбор датасетов и план MVP прорабатываются.

## Структура репозитория

Будет добавлена по мере реализации.

## Лицензия

MIT — см. [LICENSE](LICENSE).
