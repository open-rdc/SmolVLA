#!/bin/bash
# 1モデルずつ別プロセス・メモリ上限付きで実行。Windowsの空きメモリが少なければ止める。
# SigLIP は sentencepiece が要る。smolvla の環境を変えずに入れる: ~/.venvs/smolvla/bin/python -m pip install --target experiments/language/scripts/extra_pkgs sentencepiece
S=$(dirname "$0"); OUT=/mnt/d/hf_cache_encoder_test/results_v2.jsonl
for m in "$@"; do
  grep -q "\"model\": \"$m\"" "$OUT" 2>/dev/null && { echo "SKIP $m"; continue; }
  free_mb=$(powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)" | tr -d '\r')
  cfree_mb=$(powershell.exe -NoProfile -Command "[int]((Get-PSDrive C).Free/1MB)" | tr -d '\r')
  echo "== $m  (Windows空きメモリ ${free_mb}MB / C:空き ${cfree_mb}MB)"
  if [ "$free_mb" -lt 1000 ]; then echo "STOP: Windowsの空きメモリ不足"; exit 1; fi
  cache=/mnt/d/hf_cache_encoder_test; [[ $m == *SmolVLM* ]] && cache=$HOME/.cache/huggingface/hub
  HF_HUB_CACHE=$cache HF_XET_CACHE=/mnt/d/hf_cache_encoder_test/xet HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=$S/extra_pkgs \
    systemd-run --user --scope -q -p MemoryMax=2500M -p MemorySwapMax=0 \
    ~/.venvs/smolvla/bin/python "$S/encoder_lr_test_v2.py" "$m" "$OUT" "$S/../data/fisheye_tsudanuma_v2/branch_134655_ref.npz" 2>&1 | grep -vE "Warning|warn" | tail -5
done
echo DONE
