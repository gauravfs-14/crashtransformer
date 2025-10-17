# 🧾 Data Specification

CrashTransformer accepts CSV or XLSX inputs with required columns.

## Required columns

- `Crash_ID`: Unique identifier
- `Latitude`, `Longitude`: Geographic coordinates
- `CrashDate`, `DayOfWeek`, `CrashTime`: Temporal data
- `County`, `City`: Location information
- `SAE_Autonomy_Level`: Vehicle autonomy level
- `Crash_Severity`: Severity classification
- `Narrative`: Crash description text

## CSV example

```csv
Crash_ID,Latitude,Longitude,CrashDate,DayOfWeek,CrashTime,County,City,SAE_Autonomy_Level,Crash_Severity,Narrative
19955047,26.15526348,-97.99060556,2023-01-15,Sunday,14:30,Hidalgo,Weslaco,Level 0,Not Injured,"Unit 2 was stationary... Unit 1 failed to control speed and struck Unit 2 on the back end."
```

## Validation rules

- All required columns must be present (case-sensitive)
- `Crash_ID` must be non-empty; other fields should be well-formed
- `Narrative` should be plain text; keep under a few thousand characters for cost

## Tips

- Remove PII and unrelated content from narratives
- Ensure consistent date/time formats
- Large files are supported; consider batching for cost/performance
