# Aubio BeatCheck API Reference

## REST API

Base URL: `http://localhost:8000/api`

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

---

#### `GET /suites`
List available test suites.

**Response:**
```json
[
  {
    "id": "tempo",
    "name": "Tempo/Beat Tracking",
    "description": "Tests BPM detection and beat tracking"
  },
  ...
]
```

---

#### `POST /suites/{suite_id}/run`
Start analysis for a suite.

**Parameters:**
- `suite_id` (path): One of `tempo`, `onset`, `pitch`, `rhythmic`, `complex`, `all`

**Request Body:**
```json
{
  "duration": 10.0
}
```

**Response:**
```json
{
  "status": "started",
  "suite_id": "tempo"
}
```

---

#### `GET /results/{suite_id}`
Get analysis results for a suite.

**Response:**
```json
[
  {
    "signal_name": "click_120bpm",
    "status": "completed",
    "metrics": {
      "tempo_bpm": 120.0,
      "beat_count": 19,
      "evaluation": {
        "precision": 1.0,
        "recall": 0.95,
        "f_measure": 0.97,
        "mae_ms": 12.5,
        "false_positives": [],
        "false_negatives": [1.5]
      },
      "stats": {
        "mean_us": 45.2,
        "p95_us": 78.1
      }
    }
  }
]
```

---

#### `GET /results/{suite_id}/{signal_name}/plot`
Get waveform visualization as PNG.

**Response:** `image/png`

---

## CLI Artifact Formats

### `test_input.json`
```json
{
  "suite": "tempo",
  "duration": 10.0,
  "sample_rate": 44100
}
```

### `ground_truth.json`
```json
{
  "click_120bpm": {
    "category": "tempo",
    "signal_definition": {
      "signal_type": "tempo",
      "metadata": {
        "sample_rate": 44100,
        "duration": 10.0,
        "description": "Click track at 120 BPM",
        "bpm": 120.0
      },
      "ground_truth": {
        "beats": [
          {"time": 0.0, "beat_number": 1, "bar": 1},
          {"time": 0.5, "beat_number": 2, "bar": 1}
        ]
      }
    }
  }
}
```

### `analysis_results.json`
```json
{
  "suite": "tempo",
  "total_signals": 12,
  "completed": 12,
  "failed": 0,
  "signals": [
    {
      "signal_name": "click_120bpm",
      "category": "tempo",
      "status": "completed",
      "tempo_bpm": 120.0,
      "beat_count": 19,
      "detected_events": [0.0, 0.5, 1.0, ...],
      "ground_truth_events": [0.0, 0.5, 1.0, ...],
      "evaluation": {
        "precision": 1.0,
        "recall": 0.95,
        "f_measure": 0.97,
        "mean_absolute_error_ms": 12.5
      }
    }
  ]
}
```

### `evaluation.json`
```json
{
  "suite": "tempo",
  "summary": {
    "total": 12,
    "completed": 12,
    "failed": 0,
    "pass_rate": 1.0
  },
  "per_signal": {
    "click_120bpm": {
      "precision": 1.0,
      "recall": 0.95,
      "f_measure": 0.97,
      "mean_absolute_error_ms": 12.5
    }
  }
}
```

---

## Pydantic Models

### `EvaluationMetrics`
```python
class EvaluationMetrics(BaseModel):
    precision: float          # 0.0 - 1.0
    recall: float             # 0.0 - 1.0
    f_measure: float          # 0.0 - 1.0
    mean_absolute_error_ms: float  # Timing error in ms
    false_positives: list[float]   # Timestamps
    false_negatives: list[float]   # Timestamps
    matched_events: list[tuple[float, float]]  # (gt, detected)
```

### `SignalDefinition`
```python
class SignalDefinition(BaseModel):
    signal_type: Literal["tempo", "onset", "pitch", "complex"]
    metadata: SignalMetadata
    ground_truth: GroundTruth
    test_criteria: TestCriteria
```
