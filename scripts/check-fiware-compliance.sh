#!/usr/bin/env bash
# FIWARE Compliance Check (module repo version v2)
# WARNING-only for pre-existing issues; exit 1 only for severe violations.
set -euo pipefail
APP_DIR="${1:-backend/app}"

if [ ! -d "$APP_DIR" ]; then
    if [ -d "backend" ] && [ ! -d "backend/app" ]; then
        APP_DIR="backend"
    else
        echo "No Python backend found — skipping compliance check"
        exit 0
    fi
fi

severe=0
warnings=0

# Check 1 (severe): No verify=False
hits=$(grep -rn "verify\s*=\s*False" "$APP_DIR" --include="*.py" 2>/dev/null || true)
if [ -n "$hits" ]; then
    echo "❌ CRITICAL: verify=False in:"
    echo "$hits"
    severe=$((severe + 1))
fi

# Check 2 (warning): Orion-LD callers without canonical headers
# Exclude config.py (defines ORION_URL but doesn't make calls)
for f in $(grep -rlE 'ORION_URL|orion-ld-service|/ngsi-ld/v1' --include="*.py" "$APP_DIR" 2>/dev/null || true); do
    basename "$f" | grep -q "config.py" && continue
    has_canonical=$(grep -cE 'OrionClient|SyncOrionClient|inject_fiware_headers' "$f" 2>/dev/null || true)
    if [ "$has_canonical" -eq 0 ]; then
        echo "⚠️  WARNING: $f uses Orion-LD but doesn't import canonical headers"
        warnings=$((warnings + 1))
    fi
done

# Check 3 (severe): No direct INSERT INTO for entity data (exclude migrations, tests, deprecated)
hits=$(grep -rnE "INSERT\s+INTO" --include="*.py" "$APP_DIR" 2>/dev/null | grep -vE "notification_handler|subscription_manager|/tests/|/migrations/" || true)
if [ -n "$hits" ]; then
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d: -f1)
        linenum=$(echo "$line" | cut -d: -f2)
        ctx_before=$(sed -n "$((linenum - 8)),$((linenum - 1))p" "$file" 2>/dev/null)
        if ! echo "$ctx_before" | grep -qE '"""DEPRECATED'; then
            echo "❌ CRITICAL: Direct INSERT in $file:$linenum"
            severe=$((severe + 1))
        fi
    done <<< "$hits"
fi

# Check 4 (warning): Legacy ref<Type> naming
hits=$(grep -rnE '\bref[A-Z][A-Za-z]*\b' --include="*.py" "$APP_DIR" 2>/dev/null || true)
if [ -n "$hits" ]; then
    echo "⚠️  WARNING: Found legacy 'ref<Type>' naming — migrate to has<Type>:"
    echo "$hits"
    warnings=$((warnings + 1))
fi

echo ""
echo "Summary: $severe critical, $warnings warnings"
if [ "$severe" -gt 0 ]; then
    echo "❌ Blocking: $severe critical violation(s) found."
    exit 1
fi
if [ "$warnings" -gt 0 ]; then
    echo "⚠️  $warnings warnings — not blocking, but should be addressed."
fi
echo "✅ FIWARE compliance checks passed."
