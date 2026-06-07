#!/usr/bin/env bash
# ABOUTME: Preflight check for the train_wm SLURM scripts — parses a given .slurm
# ABOUTME: and verifies read access to shared inputs and write access to per-user outputs.
set -uo pipefail

# Usage: ./preflight_check.sh path/to/run_*.slurm [--no-data] [--deps]
#   By default also builds the train+val dataloader and pulls a few samples.
#   --no-data  skip the dataloader check (path/writability checks only, instant).
#   --deps     also run a pixi env import smoke test and nvidia-smi (slower).
# Run this ON THE CLUSTER (login node or interactive alloc) before sbatch.

SLURM="${1:-}"; shift || true
DEPS=0; DATA=1
for a in "$@"; do
  case "$a" in
    --deps) DEPS=1;;
    --data) DATA=1;;
    --no-data) DATA=0;;
    *) echo "unknown option: $a" >&2; exit 2;;
  esac
done

if [[ -z "$SLURM" || ! -f "$SLURM" ]]; then
  echo "usage: $0 path/to/run_*.slurm [--no-data] [--deps]" >&2
  exit 2
fi

if [[ -t 1 ]]; then R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'; else R=; G=; Y=; B=; N=; fi
PASS=0; FAIL=0; WARN=0
ok()   { PASS=$((PASS+1)); printf "  ${G}PASS${N} %s\n" "$1"; }
warn() { WARN=$((WARN+1)); printf "  ${Y}WARN${N} %s\n" "$1"; }
fail() { FAIL=$((FAIL+1)); printf "  ${R}FAIL${N} %s\n" "$1"; }
hdr()  { printf "\n${B}== %s ==${N}\n" "$1"; }

# --- path extraction from the slurm file ---------------------------------
export_var()  { grep -E "^export ${1}=" "$SLURM" | head -1 | sed -E "s/^export ${1}=//" | tr -d '"'; }
launch_arg()  { awk -v f="--$1" '/^[[:space:]]*#/{next} {for(i=1;i<=NF;i++) if($i==f){print $(i+1); exit}}' "$SLURM"; }
sbatch_log()  { grep -E '^#SBATCH[[:space:]]+--output=' "$SLURM" | head -1 | sed -E 's/^#SBATCH[[:space:]]+--output=//'; }

JOBNAME=$(grep -E '^#SBATCH[[:space:]]+--job-name=' "$SLURM" | head -1 | sed -E 's/.*--job-name=//')
LOG_PATTERN=$(sbatch_log)
LOG_DIR=$(dirname "$LOG_PATTERN")
HF_HOME=$(export_var HF_HOME)
TORCH_HOME=$(export_var TORCH_HOME)
XDG_CACHE_HOME=$(export_var XDG_CACHE_HOME)
PROJECT_DIR=$(grep -E '^cd[[:space:]]' "$SLURM" | head -1 | awk '{print $2}')

ROOT=$(launch_arg dataset_root_path)
META=$(launch_arg dataset_meta_info_path)
NAMES=$(launch_arg dataset_names)
CFGS=$(launch_arg dataset_cfgs)
STAT_CFGS=$(launch_arg dataset_stat_cfgs)
SVD=$(launch_arg svd_model_path)
CLIP=$(launch_arg clip_model_path)
CKPT=$(launch_arg ckpt_path)
OUTPUT_DIR=$(launch_arg output_dir)

printf "${B}Preflight for %s${N}  (job: %s)\n" "$SLURM" "${JOBNAME:-?}"

# --- environment ---------------------------------------------------------
hdr "environment / pixi"
if command -v pixi >/dev/null 2>&1; then ok "pixi on PATH: $(command -v pixi)"
else fail "pixi NOT on PATH (a non-interactive SLURM shell may skip .bashrc; add 'export PATH=\$HOME/.pixi/bin:\$PATH' to the script)"; fi
if [[ -n "$PROJECT_DIR" && -d "$PROJECT_DIR" ]]; then
  ok "project dir: $PROJECT_DIR"
  [[ -f "$PROJECT_DIR/pixi.toml" ]]        && ok "pixi.toml present"          || fail "pixi.toml MISSING in project dir"
  [[ -f "$PROJECT_DIR/scripts/train_wm.py" ]] && ok "scripts/train_wm.py present" || fail "scripts/train_wm.py MISSING"
else fail "project dir MISSING: ${PROJECT_DIR:-<unset>}"; fi

# --- readable shared inputs ----------------------------------------------
check_read() { # path label  (dir or file, must be readable)
  if [[ -e "$1" && -r "$1" ]]; then ok "$2: $1"
  elif [[ -e "$1" ]]; then fail "$2 exists but NOT readable: $1"
  else fail "$2 MISSING: $1"; fi
}
hdr "shared inputs (read)"
check_read "$ROOT" "dataset_root_path"
check_read "$META" "dataset_meta_info_path"
check_read "$SVD"  "svd_model_path"
check_read "$CLIP" "clip_model_path"
check_read "$CKPT" "ckpt_path (resume checkpoint)"

# --- per-cfg dataset manifests -------------------------------------------
# loader opens {meta}/{cfg}/{train,val}_sample.json and a stat.json per dataset.
hdr "dataset manifests (read)"
IFS='+' read -ra CFG_ARR  <<< "$CFGS"
IFS='+' read -ra NAME_ARR <<< "$NAMES"
IFS='+' read -ra STAT_ARR <<< "$STAT_CFGS"
for i in "${!CFG_ARR[@]}"; do
  cfg="${CFG_ARR[$i]}"; name="${NAME_ARR[$i]:-}"
  # stat_cfgs of length 1 is broadcast across all datasets by the loader.
  if   [[ ${#STAT_ARR[@]} -gt $i ]]; then statcfg="${STAT_ARR[$i]}"
  elif [[ ${#STAT_ARR[@]} -eq 1  ]]; then statcfg="${STAT_ARR[0]}"
  else statcfg=""; fi

  for mode in train val; do check_read "$META/$cfg/${mode}_sample.json" "  [$cfg] ${mode}_sample.json"; done

  # stat.json: loader tries stat_cfg, cfg, name, then meta root — first hit wins.
  found=""
  for cand in "$META/$statcfg/stat.json" "$META/$cfg/stat.json" "$META/$name/stat.json" "$META/stat.json"; do
    [[ -n "$cand" && -r "$cand" ]] && { found="$cand"; break; }
  done
  if [[ -n "$found" ]]; then ok "  [$cfg] stat.json: $found"
  else fail "  [$cfg] stat.json MISSING (tried $statcfg/, $cfg/, $name/, meta root)"; fi
done

# --- writable per-user outputs -------------------------------------------
check_write() { # dir label  (mkdir -p then touch-test)
  local d="$1" t
  if [[ -z "$d" ]]; then fail "$2 path unset"; return; fi
  if mkdir -p "$d" 2>/dev/null; then
    t="$d/.preflight_write_test.$$"
    if touch "$t" 2>/dev/null; then rm -f "$t"; ok "$2 writable: $d"
    else fail "$2 NOT writable: $d"; fi
  else fail "$2 cannot create: $d"; fi
}
hdr "per-user outputs (write)"
check_write "$LOG_DIR"        "SBATCH log dir"
check_write "$HF_HOME"        "HF_HOME"
check_write "$TORCH_HOME"     "TORCH_HOME"
check_write "$XDG_CACHE_HOME" "XDG_CACHE_HOME"
check_write "$OUTPUT_DIR"     "output_dir"

# --- optional dependency smoke test --------------------------------------
if [[ "$DEPS" -eq 1 ]]; then
  hdr "pixi env deps (--deps)"
  if command -v pixi >/dev/null 2>&1 && [[ -f "$PROJECT_DIR/pixi.toml" ]]; then
    if (cd "$PROJECT_DIR" && pixi run python -c "import torch,accelerate,diffusers,transformers" 2>/dev/null); then
      ok "imports torch/accelerate/diffusers/transformers"
    else fail "pixi env import failed (run 'pixi install' in $PROJECT_DIR)"; fi
  else warn "skipped import test (pixi or pixi.toml unavailable)"; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    [[ "$gpus" -ge 8 ]] && ok "nvidia-smi sees $gpus GPUs (need 8)" || warn "nvidia-smi sees $gpus GPUs (job requests 8 — run inside the alloc)"
  else warn "nvidia-smi not found (run inside a GPU alloc to check)"; fi
fi

# --- dataset / dataloader smoke test (default; --no-data to skip) --------
if [[ "$DATA" -eq 1 ]]; then
  hdr "dataset dataloader"
  if command -v pixi >/dev/null 2>&1 && [[ -f "$PROJECT_DIR/verify_dataset.py" ]]; then
    abs_slurm="$(cd "$(dirname "$SLURM")" && pwd)/$(basename "$SLURM")"
    if (cd "$PROJECT_DIR" && pixi run python verify_dataset.py "$abs_slurm"); then
      ok "built train+val dataloader and pulled samples"
    else fail "dataset verifier failed (see output above)"; fi
  else warn "skipped dataset verify (pixi or $PROJECT_DIR/verify_dataset.py unavailable)"; fi
fi

# --- summary -------------------------------------------------------------
printf "\n${B}summary:${N} ${G}%d pass${N}, ${Y}%d warn${N}, ${R}%d fail${N}\n" "$PASS" "$WARN" "$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1 || exit 0
