# Pipeline Logging Guide

## Overview
The pipeline now automatically logs all execution details to timestamped log files in the `logs/` directory.

## Log File Location
- **Directory**: `logs/`
- **Filename Format**: `pipeline_YYYY-MM-DD_HH-MM-SS.log`
- **Example**: `logs/pipeline_2026-01-18_14-32-45.log`

## What Gets Logged

Each pipeline run captures:
- **Start time** - When the pipeline began
- **Step execution** - Each pipeline step with start/end times
- **Execution time** - How long each step took
- **Errors** - Any errors encountered with exit codes
- **Completion status** - Success or failure

## Log Levels

### File Logs (Detailed)
- **DEBUG**: Timestamps, execution times, technical details
- **INFO**: Step progress, completion status
- **ERROR**: Errors and failures

### Console Output (User-Friendly)
- **INFO**: Only high-level progress messages
- **ERROR**: Only error messages

## Example Log Output

```
2026-01-18 14:32:45 | INFO     | 🚀 ================================================================================
2026-01-18 14:32:45 | INFO     |    CUSTOMER CHURN PREDICTION & CLV ANALYSIS - COMPLETE PIPELINE
2026-01-18 14:32:45 | INFO     | ================================================================================
2026-01-18 14:32:45 | INFO     | Pipeline started at: 2026-01-18 14:32:45
2026-01-18 14:32:45 | INFO     | Logs will be saved to: C:\...\Project\logs\pipeline_2026-01-18_14-32-45.log
2026-01-18 14:32:45 | INFO     | 
2026-01-18 14:32:45 | INFO     | ================================================================================
2026-01-18 14:32:45 | INFO     |   Step 1/5: DATA PREPARATION (load, clean, engineer features, split)
2026-01-18 14:32:45 | DEBUG    | Start time: 2026-01-18 14:32:45
2026-01-18 14:35:12 | DEBUG    | Elapsed time: 147.32 seconds
2026-01-18 14:35:12 | INFO     | ✅ Step 1/5 completed successfully
```

## Viewing Logs

### Option 1: Open in Text Editor
```bash
# Windows PowerShell
notepad logs/pipeline_2026-01-18_14-32-45.log
```

### Option 2: View in Terminal
```bash
# Windows PowerShell
Get-Content logs/pipeline_2026-01-18_14-32-45.log

# Git Bash
cat logs/pipeline_2026-01-18_14-32-45.log

# View last 50 lines
Get-Content logs/pipeline_2026-01-18_14-32-45.log -Tail 50
```

### Option 3: Search Logs
```bash
# Search for errors
Select-String "ERROR" logs/pipeline_2026-01-18_14-32-45.log

# Search for a specific step
Select-String "Step 3" logs/pipeline_2026-01-18_14-32-45.log
```

## Usage

Simply run the pipeline as usual:
```bash
python run_pipeline.py
```

Logs will be automatically created and saved!

## Troubleshooting

### "logs directory doesn't exist"
- The directory is created automatically on the first run. No action needed.

### "Permission denied" error
- Ensure you have write permissions in the project directory
- Try running as administrator if on Windows

### Viewing old logs
- All logs are timestamped and preserved
- Check the `logs/` directory for previous runs
- Useful for tracking pipeline history and debugging

## Tips

1. **Keep logs for reference** - They help track pipeline performance over time
2. **Check logs for timing** - See which steps are slowest
3. **Debug failures** - File logs contain detailed error information
4. **Monitor progress** - Use logs to verify each step completed successfully
