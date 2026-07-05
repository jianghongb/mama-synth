#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -J download_ambl
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/download_ambl_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/download_ambl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Download Advanced-MRI-Breast-Lesions from TCIA to Berzelius.
# Downloads pre-contrast (MASK) + MultiPhase (DCE) + ROI (seg) series.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show tcia_utils > /dev/null 2>&1 || pip install tcia_utils

OUTPUT=$PROJ/Advanced-MRI-Breast-Lesions

python -c "
from tcia_utils import nbia
import pandas as pd
from pathlib import Path

output_dir = '$OUTPUT'
Path(output_dir).mkdir(parents=True, exist_ok=True)

print('Fetching series metadata...')
series = nbia.getSeries(collection='Advanced-MRI-Breast-Lesions')
df = pd.DataFrame(series)
print(f'Total series: {len(df)}, Patients: {df[\"PatientID\"].nunique()}')

# Select relevant series: MASK (pre), MultiPhase (DCE), ROI (seg)
relevant = df[df['SeriesDescription'].isin([
    'AX Sen Vibrant MASK',
    'AX Sen Vibrant MultiPhase',
    'ROI'
])]
print(f'Relevant series to download: {len(relevant)}')
print(relevant['SeriesDescription'].value_counts())

# Download all relevant series
uids = relevant['SeriesInstanceUID'].tolist()
print(f'Starting download of {len(uids)} series...')
nbia.downloadSeries(uids, input_type='list', path=output_dir, max_workers=16)
print('Download complete!')
"

echo "=== Download finished ==="
echo "Files:"
find $OUTPUT -name "*.dcm" | wc -l
echo "Total size:"
du -sh $OUTPUT
