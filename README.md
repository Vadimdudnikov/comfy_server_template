# Comfy Server Template

Один файл `app/workflow.json` — меняете его и раскатываете любой pipeline.

## Новый pipeline за 2 шага

**1.** ComfyUI → **Save (API Format)** → вставить в `app/workflow.json`

**2.** Дописать секции сервиса одной командой:

```bash
python scripts/inspect_workflow.py --init
```

Скрипт добавит в **тот же файл** внизу все редактируемые поля workflow:

```json
"_inputs": {
  "45_text":  {"node": "45", "field": "text"},
  "44_seed":  {"node": "44", "field": "seed"},
  "41_width": {"node": "41", "field": "width"}
},
"_output": {"node": "9"},
"_models": {}
```

Имена `{node_id}_{field}` — автоматически, без привязки к типу pipeline.
Удалите лишнее и переименуйте нужные ключи (например `45_text` → `prompt`).

При смене pipeline — снова вставляете экспорт и `--init --force`.

## Структура workflow.json

```
┌─────────────────────────────────┐
│  "9": { ... },   ← узлы ComfyUI │  стандартный экспорт, как есть
│  "44": { ... },                 │
│  ...                            │
├─────────────────────────────────┤
│  "_inputs": { ... },            │  что менять через API
│  "_output": { "node": "9" },    │  откуда брать картинку
│  "_models": { ... }             │  автозагрузка (опционально)
└─────────────────────────────────┘
```

ComfyUI секции `_` не трогает — это метаданные только для нашего сервиса.

## Как читать _inputs

```json
"45_text": {"node": "45", "field": "text"}
```

→ параметр `45_text` из HTTP-запроса попадёт в узел `45`, поле `text`.
Ключи можно переименовать как угодно — важны только `node` и `field`.

```bash
python scripts/inspect_workflow.py   # таблица узлов и всех полей
```

## Запуск

```bash
./start.sh
```

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"45_text": "a cat", "44_seed": 42}, "wait": true}'
```

После замены JSON: `curl -X POST http://localhost:8000/workflow/reload`

## _models (опционально)

```json
"_models": {
  "vae": {
    "ae.safetensors": {
      "url": "https://huggingface.co/...",
      "path": "models/vae/ae.safetensors",
      "description": "VAE"
    }
  }
}
```
