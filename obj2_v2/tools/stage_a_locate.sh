#!/usr/bin/env bash
# stage_a_locate.sh  (obj2_v2, stage A step 1, 7 Nov 2026)
#
# What it does
#   Read-only. Lists every copy under your home folder of the stage-2 (Objective 2) code files that
#   trained and ran the old model, with date, md5 and size, so we pick the copy that made the FIXAB
#   model before anything is copied into obj2_v2/original. Also lists the stage2_outputs* folders
#   (checkpoints) and the log files inside them, needed for audit finding F11 (which training phase
#   epoch 149 was in). Writes only the log file.
#
# Why
#   The project holds copies dated 13 May to 29 Jul 2026; your machine may hold later versions
#   (as found for Objective 1 on 24 Oct). The audit must run on the code that made the model.
#
# Run (from PSD_Dispersal_pool/obj2_v2/work)
#   bash ../tools/stage_a_locate.sh ../logs/stage_a_locate_7nov.txt
#
# Output
#   Part 1: one line per copy: date  md5  size  path, grouped by file name, with a count per name.
#   Part 2: stage2_outputs* folders with their .pt and log files (date, size, path).
OUT="${1:-../logs/stage_a_locate_7nov.txt}"
ROOT="${STAGE_A_ROOT:-$HOME}"
NAMES="training.py run_training.py training_v2_patch.py data_preprocessing.py data_preprocessing_v2_patch.py \
interaction_encoder_v2_patch.py config.py ecodiffusion.py ecodiffusion_spatial_cond.py ecodiffusion_sample_spatial.py \
temporal_encoder.py env_encoder.py interaction_encoder.py unet.py diffusion.py fix_a_b_extrapolation.py \
generate_reconstructions_spatial.py generate_reconstructions_proportional.py proportional_observations.py \
baseline_smoother_compare.py posterior_per_species.py"
{
  echo "stage_a_locate.sh run $(date '+%Y-%m-%d %H:%M') under $ROOT"
  echo "== part 1: copies of the stage-2 code files (date  md5  size  path)"
  for f in $NAMES; do
    n=0
    while IFS= read -r -d '' p; do
      case "$f" in config.py) grep -q "class TrainingConfig" "$p" 2>/dev/null || continue ;; esac
      printf '%s  %s  %9s  %s\n' "$(date -r "$p" '+%Y-%m-%d %H:%M')" "$(md5sum < "$p" | cut -c1-32)" "$(stat -c %s "$p")" "$p"
      n=$((n + 1))
    done < <(find "$ROOT" -type f -name "$f" -not -path '*/.*' -not -path '*site-packages*' -not -path '*/obj2_v2/*' -print0 2>/dev/null | sort -z)
    echo "-- $f: $n copies"
  done
  echo "== part 2: checkpoint folders (stage2_outputs*) and their .pt and log files"
  while IFS= read -r -d '' d; do
    echo "folder $d"
    while IFS= read -r -d '' p; do
      printf '   %s  %12s  %s\n' "$(date -r "$p" '+%Y-%m-%d %H:%M')" "$(stat -c %s "$p")" "${p#$d/}"
    done < <(find "$d" -maxdepth 3 -type f \( -name '*.pt' -o -name '*.log' -o -name '*.txt' -o -name '*.json' -o -name 'nohup*' \) -print0 2>/dev/null | sort -z)
  done < <(find "$ROOT" -type d -name 'stage2_outputs*' -not -path '*/.*' -print0 2>/dev/null | sort -z)
  echo "== end"
} | tee "$OUT"
