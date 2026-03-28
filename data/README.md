# Data Layout

The project uses a deterministic synthetic medical-notes dataset so the
format stays consistent and the refusal cases are fully controlled.

Run:

```powershell
py .\prepare_dataset.py --num-examples 4000 --output-dir .\data\processed
```

Files produced:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `dataset_card.json`

Each row contains:

- `id`
- `input_text`
- `schema_hint`
- `target_json`
- `can_extract`
- `domain`

The JSON schema is stable across all examples:

```json
{
  "can_extract": true,
  "refusal_reason": null,
  "record": {
    "patient_name": "string",
    "age": 42,
    "sex": "female",
    "encounter_date": "2026-01-14",
    "chief_complaint": "chest tightness",
    "diagnoses": ["hypertension"],
    "medications": [
      {
        "name": "lisinopril",
        "dose": "10 mg daily"
      }
    ],
    "allergies": ["penicillin"],
    "vitals": {
      "blood_pressure": "142/88",
      "heart_rate": 84,
      "temperature_f": 98.4
    },
    "follow_up_days": 14
  }
}
```

Refusal examples set `can_extract=false`, `record=null`, and provide a
short `refusal_reason`.
