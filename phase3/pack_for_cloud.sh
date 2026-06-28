#!/usr/bin/env bash
# Package the MINIMAL files needed to fine-tune on a cloud GPU (Colab/Modal) into one tarball.
# Fine-tune only needs: the backbone, labeled train images + csv, and the phase3 code.
# (The 144k unlabeled / 118GB dataset is NOT needed for the labeled fine-tune.)
#
#   bash phase3/pack_for_cloud.sh                 # labeled-only (~0.8 GB)
#   bash phase3/pack_for_cloud.sh with-negs 6000  # + 6000 unlabeled CONFIDENT_NEGATIVE images (~1.4 GB)
set -e
cd "$(dirname "$0")/.."
OUT=rare26_finetune.tar.gz
STAGE=$(mktemp -d)/rare26
mkdir -p "$STAGE/phase3" "$STAGE/dataset"

echo "staging code + backbone + labeled train..."
cp phase3/*.py "$STAGE/phase3/"
cp dinov2.pth "$STAGE/"
cp dataset/train.csv "$STAGE/dataset/"
cp -R dataset/train "$STAGE/dataset/train"

if [ "$1" = "with-negs" ]; then
  N=${2:-6000}
  echo "staging $N unlabeled CONFIDENT_NEGATIVE images..."
  mkdir -p "$STAGE/dataset/unl_neg"
  head -n "$N" phase3/cache/unl_confneg.txt | while read -r p; do cp "$p" "$STAGE/dataset/unl_neg/" 2>/dev/null || true; done
  ( cd "$STAGE/dataset/unl_neg" && ls | sed 's#^#dataset/unl_neg/#' > ../unl_neg.txt )
fi

echo "compressing -> $OUT ..."
tar -czf "$OUT" -C "$(dirname "$STAGE")" rare26
rm -rf "$(dirname "$STAGE")"
echo "DONE: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "Upload $OUT to Google Drive, then run phase3/colab_finetune.ipynb"
