# Flywheel scheduling

The "nightly cron" mentioned in `scripts/flywheel.py` has never actually been
installed — the flywheel is manually cranked. This doc + `scripts/flywheel_nightly.sh`
make it schedulable. **Nothing here self-installs**; run the install step
yourself when you want it live.

## What a nightly cycle does

`scripts/flywheel_nightly.sh`:

1. `pipeline pack doctor --no-smoke` — skip the night early if a pack is
   structurally broken.
2. One `scripts/flywheel.py` cycle per ready pack (compose → autosearch →
   graded ledger row). Edit the pack list in the script as packs become
   candidate+search ready.
3. `pipeline web` — regenerate the static dashboard.

Logs land in `data/pipeline/nightly/<stamp>.log`.

Interruption is cheap: compose runs resume with `--resume <run_id>` and
autosearch reuses its `evals.json` cache, so a sleep/kill mid-cycle doesn't
re-pay codex spend.

## Install (launchd, macOS)

Save as `~/Library/LaunchAgents/com.congruity.flywheel.plist` (adjust the
repo path), then `launchctl load ~/Library/LaunchAgents/com.congruity.flywheel.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.congruity.flywheel</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>/path/to/congruity/scripts/flywheel_nightly.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>2</integer>
    <key>Minute</key><integer>17</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/path/to/congruity/data/pipeline/nightly/launchd.out</string>
  <key>StandardErrorPath</key>
  <string>/path/to/congruity/data/pipeline/nightly/launchd.err</string>
</dict>
</plist>
```

`launchctl unload …` to stop. (cron works too: `17 2 * * * /bin/zsh /path/to/congruity/scripts/flywheel_nightly.sh`.)

## Cost note

One cycle ≈ one compose run + one autosearch per composition with a recipe.
Watch `llm_usage.json` per run and the funnel report
(`scripts/funnel_report.py`) before increasing frequency or pack count.
