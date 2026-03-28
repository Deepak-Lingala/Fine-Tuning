from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NAMES = [
    "Ava Johnson",
    "Noah Martinez",
    "Mia Patel",
    "Liam Chen",
    "Sophia Williams",
    "Ethan Garcia",
    "Olivia Nguyen",
    "Lucas Brown",
    "Emma Lopez",
    "Elijah Davis",
]

SEXES = ["female", "male"]
COMPLAINTS = [
    "shortness of breath",
    "persistent cough",
    "headache",
    "abdominal pain",
    "fatigue",
    "chest tightness",
    "dizziness",
    "low back pain",
]
DIAGNOSES = [
    "hypertension",
    "viral upper respiratory infection",
    "migraine",
    "type 2 diabetes",
    "asthma exacerbation",
    "gastroenteritis",
    "anxiety",
    "lumbar strain",
]
MEDICATIONS = [
    ("lisinopril", "10 mg daily"),
    ("metformin", "500 mg twice daily"),
    ("albuterol", "2 puffs as needed"),
    ("ibuprofen", "400 mg every 8 hours as needed"),
    ("sertraline", "50 mg daily"),
    ("ondansetron", "4 mg every 8 hours as needed"),
]
ALLERGIES = ["penicillin", "sulfa", "latex", "none known", "peanuts"]
REFUSAL_TEXTS = [
    "Marketing brainstorm notes for next quarter campaign. No patient details included.",
    "Shopping list: bananas, oat milk, detergent, printer ink. Need to reorder by Friday.",
    "Project retro summary: team velocity improved after reducing weekly meetings.",
    "Travel itinerary draft with flight numbers and hotel check-in instructions.",
]


@dataclass
class Example:
    id: str
    input_text: str
    schema_hint: str
    target_json: dict[str, Any]
    can_extract: bool
    domain: str = "medical_notes"

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "input_text": self.input_text,
                "schema_hint": self.schema_hint,
                "target_json": self.target_json,
                "can_extract": self.can_extract,
                "domain": self.domain,
            },
            ensure_ascii=True,
        )


def _build_note(rng: random.Random, idx: int) -> Example:
    name = rng.choice(NAMES)
    age = rng.randint(18, 84)
    sex = rng.choice(SEXES)
    complaint = rng.choice(COMPLAINTS)
    diagnoses = rng.sample(DIAGNOSES, k=rng.randint(1, 2))
    meds = rng.sample(MEDICATIONS, k=rng.randint(1, 2))
    allergies = rng.sample(ALLERGIES, k=1)
    systolic = rng.randint(108, 156)
    diastolic = rng.randint(68, 98)
    heart_rate = rng.randint(62, 112)
    temperature = round(rng.uniform(97.6, 100.8), 1)
    follow_up_days = rng.choice([7, 10, 14, 21, 30])
    encounter_date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    note = (
        f"ED/Clinic note :: Pt {name}, {age} y/o {sex}. "
        f"Seen on {encounter_date} for {complaint}. "
        f"Assessment mentions {', '.join(diagnoses)}. "
        "Current meds: "
        + "; ".join(f"{med_name} {dose}" for med_name, dose in meds)
        + ". "
        f"Allergies: {', '.join(allergies)}. "
        f"Vitals -> BP {systolic}/{diastolic}, HR {heart_rate}, Temp {temperature}F. "
        f"Follow-up requested in {follow_up_days} days. "
        "Transcribed from mixed shorthand and dictated fragments."
    )

    target = {
        "can_extract": True,
        "refusal_reason": None,
        "record": {
            "patient_name": name,
            "age": age,
            "sex": sex,
            "encounter_date": encounter_date,
            "chief_complaint": complaint,
            "diagnoses": diagnoses,
            "medications": [{"name": med_name, "dose": dose} for med_name, dose in meds],
            "allergies": allergies,
            "vitals": {
                "blood_pressure": f"{systolic}/{diastolic}",
                "heart_rate": heart_rate,
                "temperature_f": temperature,
            },
            "follow_up_days": follow_up_days,
        },
    }

    return Example(
        id=f"medical-{idx:05d}",
        input_text=note,
        schema_hint="medical_intake_v1",
        target_json=target,
        can_extract=True,
    )


def _build_refusal(rng: random.Random, idx: int) -> Example:
    return Example(
        id=f"refusal-{idx:05d}",
        input_text=rng.choice(REFUSAL_TEXTS),
        schema_hint="medical_intake_v1",
        target_json={
            "can_extract": False,
            "refusal_reason": "No extractable medical facts present.",
            "record": None,
        },
        can_extract=False,
    )


def build_dataset(num_examples: int, seed: int = 42, refusal_ratio: float = 0.12) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    refusal_count = int(num_examples * refusal_ratio)
    extract_count = num_examples - refusal_count
    for idx in range(extract_count):
        examples.append(_build_note(rng, idx))
    for idx in range(refusal_count):
        examples.append(_build_refusal(rng, idx))
    rng.shuffle(examples)
    return examples


def split_dataset(examples: list[Example], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, list[Example]]:
    train_end = int(len(examples) * train_ratio)
    val_end = train_end + int(len(examples) * val_ratio)
    return {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }


def write_jsonl(examples: list[Example], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.to_json() + "\n")


def write_dataset_card(output_dir: Path, num_examples: int, seed: int) -> None:
    card = {
        "dataset_name": "synthetic_medical_json_extraction",
        "num_examples": num_examples,
        "seed": seed,
        "schema": "medical_intake_v1",
        "refusal_policy": "Return can_extract=false when input is non-medical or lacks extractable facts.",
    }
    with (output_dir / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(card, handle, indent=2)
