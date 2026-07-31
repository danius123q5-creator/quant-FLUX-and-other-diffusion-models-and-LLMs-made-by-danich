#!/bin/bash
# Сборка двух GGUF одним базовым типом Q4_0, режим quality, разные аллокаторы.
cd /d/ComfyBot/xquant
M="D:/Comfy/models/checkpoints/flux1-dev.safetensors"
export PYTHONIOENCODING=utf-8 XQUANT_SMART=1 XQUANT_SMART_MODE=quality
for A in v1 v2; do
  export XQUANT_ALLOC=$A
  echo "=== $(date +%H:%M:%S) СТАРТ $A ==="
  python xquant_standalone.py "$M" Q4_0 > "ab_test/log_$A.txt" 2>&1
  echo "=== $(date +%H:%M:%S) ГОТОВ $A (код $?) ==="
  for f in "D:/Comfy/models/checkpoints/"*Q4_0*.gguf; do
    [ -f "$f" ] && mv -f "$f" "ab_test/flux-Q4_0-quality-$A.gguf" && echo "  -> ab_test/flux-Q4_0-quality-$A.gguf"
  done
done
ls -la ab_test/*.gguf 2>/dev/null
