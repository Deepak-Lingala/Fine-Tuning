SYSTEM_PROMPT = """You extract structured medical data from noisy notes.
Return exactly one compact JSON object and nothing else.
Always use this schema:
{
  "can_extract": boolean,
  "refusal_reason": string or null,
  "record": {
    "patient_name": string,
    "age": integer,
    "sex": string,
    "encounter_date": string,
    "chief_complaint": string,
    "diagnoses": [string],
    "medications": [{"name": string, "dose": string}],
    "allergies": [string],
    "vitals": {
      "blood_pressure": string,
      "heart_rate": integer,
      "temperature_f": number
    },
    "follow_up_days": integer
  } or null
}

If the note does not contain extractable medical facts, respond with:
{"can_extract": false, "refusal_reason": "...", "record": null}
"""


def build_user_prompt(input_text: str, schema_hint: str) -> str:
    return (
        "Extract the medical record into the required JSON schema.\n"
        f"Schema hint: {schema_hint}\n"
        "Input note:\n"
        f"{input_text}"
    )
