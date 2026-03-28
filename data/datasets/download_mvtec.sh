#!/bin/bash

set -e

OFFICIAL_URL="https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads"
DATA_DIR="data/raw"
TMP_DIR="data/tmp_mvtec"
ARCHIVE_PATH="${1:-}"

CATEGORIES=("screw" "metal_nut" "bottle")

echo " MVTec AD Official Download Extractor"
echo " Official page: $OFFICIAL_URL"
echo " Target dir   : $DATA_DIR"
echo " Categories   : ${CATEGORIES[*]}"
echo ""

if [ -z "$ARCHIVE_PATH" ]; then
    echo "[ERROR] No archive path provided."
    echo ""
    echo "1. Download the official MVTec AD archive from:"
    echo "   $OFFICIAL_URL"
    echo ""
    echo "2. Then run:"
    echo "   ./scripts/download_mvtec.sh /path/to/mvtec_anomaly_detection.tar.xz"
    echo ""
    echo "You can also use a .zip archive if that is what you downloaded."
    exit 1
fi

if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "[ERROR] Archive not found: $ARCHIVE_PATH"
    exit 1
fi

mkdir -p "$DATA_DIR"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "[INFO] Extracting archive: $ARCHIVE_PATH"

case "$ARCHIVE_PATH" in
    *.tar.xz)
        tar -xf "$ARCHIVE_PATH" -C "$TMP_DIR"
        ;;
    *.zip)
        unzip -q "$ARCHIVE_PATH" -d "$TMP_DIR"
        ;;
    *)
        echo "[ERROR] Unsupported archive format."
        echo "Use an official .tar.xz or .zip archive."
        exit 1
        ;;
esac

ROOT_FOUND=$(find "$TMP_DIR" -type d \( -name "screw" -o -name "metal_nut" -o -name "bottle" \) | head -n 1)

if [ -z "$ROOT_FOUND" ]; then
    echo "[ERROR] Could not find MVTec category folders after extraction."
    echo "[ERROR] Check that the archive came from the official MVTec AD download page."
    exit 1
fi

BASE_DIR=$(dirname "$ROOT_FOUND")

for category in "${CATEGORIES[@]}"; do
    if [ -d "$BASE_DIR/$category" ]; then
        echo "[INFO] Copying $category ..."
        rm -rf "$DATA_DIR/$category"
        cp -r "$BASE_DIR/$category" "$DATA_DIR/"
    else
        echo "[WARN] Category not found: $category"
    fi
done

echo ""
echo "[DONE] Official MVTec categories extracted to: $DATA_DIR"
echo "[DONE] Example:"
echo "       data/raw/screw/train/good"
echo "       data/raw/screw/test/good"
echo "       data/raw/screw/ground_truth"