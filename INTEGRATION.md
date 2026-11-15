# Guard-main integration contract

This module exposes HTTP API compatible with `guard-main/lib/analysis-client.ts`.

## Endpoints

- `POST /v1/jobs` — поставить анализ в очередь (рекомендуется): `202 {job_id, status: "queued"}`
- `GET /v1/jobs/{job_id}` — статус `queued` (ждёт в очереди) → `processing` (анализируется) → `completed` / `failed`; при `completed` — поле `result` (формат как у `/v1/analyze`)
- `POST /v1/analyze` — синхронный анализ (соединение держится до конца)
- `DELETE /v1/documents/{document_id}` — удалить фрагменты работы из базы (`204`)
- `GET /v1/queues` — состояние очередей по вузам
- `GET /health` — `{ready: true}`, когда модели загружены
- `Content-Type: application/json`
- Optional header: `X-API-Key` (required only if `ANALYSIS_API_KEY` is set)

## Request body

```json
{
  "content": "text to analyze",
  "filename": "document.txt",
  "document_id": 123,
  "university_id": "bsuir"
}
```

- `content` is required, non-empty string: текст или файл `FILE_BASE64|docx|<base64>` / `FILE_BASE64|pdf|<base64>`.
  DOCX читается вместе с таблицами. `.doc`, другие форматы и PDF без текстового слоя (скан) → `400` с понятным сообщением
- `category` optional (default: `uncategorized`) — тип работы; курсовые и дипломы сравниваются между собой, `seo` — со всеми категориями вуза
- `user_id` optional — автор; его другие работы исключаются из сравнения
- `exclude_document_ids` optional — дополнительно исключаемые работы (до 500)
- `filename` optional, defaults to `document.txt`
- `document_id` optional — сохраняется в payload чанков Qdrant и используется
  для атрибуции источника в `semantic_matches`
- `university_id` optional (default: `default`) — очередь анализа: 1 университет =
  1 очередь, создаётся динамически при первом запросе. Планировщик обходит очереди
  по кругу (round-robin), поэтому поток документов одного университета не блокирует
  остальные. Состояние очередей: `GET /v1/queues` (тот же `X-API-Key`).

## Success response (200)

```json
{
  "plagiarism_percent": 12.34,
  "template_percent": 3.1,
  "ai_percent": 7.89,
  "semantic_matches": [
    {
      "document_id": 42,
      "filename": "kursovaya_ivanov.docx",
      "matched_chunks": 7,
      "max_score": 0.83,
      "max_lexical_score": 0.95,
      "paraphrase_score": 0.22,
      "match_type": "exact",
      "sample": "первые 300 символов совпавшего чанка…"
    }
  ],
  "by_type": { "exact": 1, "paraphrase": 0, "semantic": 0 }
}
```

- `match_type` — каскадная классификация источника (источник истины — `cascade.py`;
  локальный каскад guard-main — только фолбэк): `exact` (лексический косинус ≥
  `LEXICAL_SCORE_THRESHOLD`, дословное копирование) → `paraphrase` (Жаккар 3-грамм
  смысловых слов ≥ `PARAPHRASE_THRESHOLD`, переформулировка) → `semantic`
  (только dense-сигнал: глубокий рерайт).
- `by_type` — число источников в каждом тире каскада.
- `semantic_matches[].fragments` — подсветка: пары `query_text` (кусок проверяемой
  работы, до 500 символов) / `source_text` (кусок источника) со `score` и `match_type`.
  Ровно один фрагмент на каждый совпавший кусок и источник, **без ограничения по
  количеству** (до исправления: максимум 20 на источник, текст обрезался до 400
  символов, dense- и lexical-сигнал одного куска давали два фрагмента).
- `template_percent` — доля кусков с **шаблонными фразами**: кусок похож на куски из
  `TEMPLATE_MIN_SOURCES` (5) и более разных работ, и ни одна из них не совпадает с
  проверяемой хотя бы по `STRONG_SOURCE_MIN_CHUNKS` (3) кускам. Типичные случаи: шапка
  вуза, «О себе» в резюме, типовое введение по методичке. Такие куски **не входят** в
  `plagiarism_percent` и не попадают в `semantic_matches`. Настоящая копия (источник
  совпадает по ≥3 кускам) засчитывается, даже если ту же работу уже скопировали многие.
- **Процент заимствований для отчёта берите из `plagiarism_percent`**, а не
  пересчитывайте по `fragments`: один кусок может совпасть с несколькими
  источниками, и сумма по источникам больше реального процента.

- `semantic_matches` — совпадения, агрегированные по работам-источникам. Поиск
  гибридный: `max_score` — максимальная косинусная близость эмбеддингов (ловит
  глубокий рерайт), `max_lexical_score` — максимальный лексический косинус
  BM25-подобных разреженных векторов в [0..1] (ловит дословное копирование;
  1.0 — точная копия чанка). Отсортированы по числу совпавших чанков; `sample` —
  фрагмент проверяемого текста. `document_id` источника может быть `null`,
  если источник индексировался без id.

## Error responses

- `400` failed to extract text from base64 file (broken PDF/DOCX)
- `401` invalid API key
- `422` invalid request schema
- `503` worker not initialized
- `500` internal worker/runtime error

## Environment variables

- `QDRANT_HOST` (default: `localhost`)
- `QDRANT_PORT` (default: `6333`)
- `QDRANT_COLLECTION` (default: `university_docs`)
- `QDRANT_API_KEY` — ключ доступа к Qdrant; в docker-compose **обязателен** (задаётся в `.env`), порт 6333 открыт только на `127.0.0.1`
- `ANALYSIS_JOB_TTL_SEC` (default: `172800`) — сколько хранится завершённая задача `/v1/jobs`
- `ANALYSIS_API_KEY` (optional; must match `ANALYSIS_SERVICE_API_KEY` on guard-main)
- `DEVICE` (default: `auto`) — `auto` | `cpu` | `cuda` (use `cuda` only with GPU-enabled PyTorch + GPU access)
- `ANALYSIS_CONCURRENCY` (default: `1`) — сколько задач анализа выполняется одновременно (очереди университетов обходятся по кругу). Внутри одного процесса анализ дополнительно сериализуется локом ML-воркера, поэтому значения >1 дают эффект только при нескольких экземплярах сервиса
- `QDRANT_SCORE_THRESHOLD` (default: `0.80`) — порог косинуса эмбеддингов (семантический сигнал)
- `LEXICAL_SCORE_THRESHOLD` (default: `0.60`) — порог лексического косинуса (дословное копирование)
- `PARAPHRASE_THRESHOLD` (default: `0.15`) — порог Жаккара смысловых 3-грамм для тира «перефраз» каскада
- `TEMPLATE_MIN_SOURCES` (default: `5`) — с какого числа разных работ похожий кусок считается шаблонной фразой; `0` — выключить правило
- `STRONG_SOURCE_MIN_CHUNKS` (default: `3`) — сколько кусков должно совпасть с одной работой, чтобы она считалась настоящим источником (её куски не бывают шаблоном)
- `MATCH_CANDIDATES` (default: `20`) — сколько кандидатов Qdrant берёт на кусок (не меньше `TEMPLATE_MIN_SOURCES`)
- `AI_ENABLED` (default: `true`) — `false` не грузит языковую модель и отдаёт `ai_percent = null` (нет сигнала)
- `AI_DETECT_BATCH_SIZE` (default: `8`) — размер батча детектора ИИ (больше — быстрее, но выше пик памяти)
- `AI_DETECT_D0` (default: `0.8`), `AI_DETECT_SCALE` (default: `1.0`) — калибровка сигмоиды (одинаково в коде и docker-compose). Цель: обычные работы ~10–30%, 60%+ — редкость. Прежние `-2.2` / `3` давали 75–100% почти любому тексту. Откалибровано по двум человеческим дипломам (2020 и 2022 гг., до ChatGPT): 9% и 26%. Перепроверять по работам с известным авторством (человек / ИИ)
- `AI_CHUNK_SIZE` / `AI_CHUNK_OVERLAP` (default: `1000` / `100`) — окно детектора, отдельно от поиска (поиск всегда 500/50)
- `AI_MIN_CHUNK_CHARS` (default: `100`) — окна этой длины и короче пропускаются
- `AI_MAX_TOKENS` (default: `1500`) — обрезка окна
- `AI_AGGREGATION` (default: `mean`) — `mean` | `median` | `max` | `top_share`
- `AI_TOP_SHARE_THRESHOLD` (default: `AI_DETECT_D0`) — порог для `top_share`
- `AI_DEBUG_LOG` / `AI_DEBUG_LOG_CHUNKS` (default: `false`) — сводка и окна в stdout
- `EMBEDDING_MODEL` (default: `BAAI/bge-m3`), `AI_MODEL` (default: `sberbank-ai/rugpt3small_based_on_gpt2`) — замена моделей; `AI_MODEL` требует пересборки образа, остальные `AI_*` — только перезапуск
- `CORS_ALLOW_ORIGINS` (optional) — список origin через запятую для браузерных запросов напрямую к API
- `TORCH_CPU_THREADS` (optional) — OpenMP/torch CPU threads on CPU runs (default: up to 8 or available cores)

## Guard-main pairing

- ML base URL on web: **`ANALYSIS_SERVICE_URL`** (e.g. `http://172.16.251.219:8765`, no trailing slash).
- Shared secret header: **`ANALYSIS_SERVICE_API_KEY`** on Next.js MUST equal **`ANALYSIS_API_KEY`** on this service.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8765
```

Или в Docker:

```bash
docker compose up -d --build analysis
```