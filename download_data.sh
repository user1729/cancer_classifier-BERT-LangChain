#!/bin/bash
#view url: https://drive.google.com/file/d/1DVPiNx-UKO7B6HvNniGIOOadJ3yq083i/view?usp=sharing
FILE_ID="1DVPiNx-UKO7B6HvNniGIOOadJ3yq083i"
FILE_NAME="cancer_data.zip"

echo "Downloading file from Google Drive..."

curl -L -o "$FILE_NAME" "https://docs.google.com/uc?export=download&id=${FILE_ID}"

# Check download success
if [ ! -f "$FILE_NAME" ]; then
  echo "Download failed!"
  exit 1
fi

# Unzip
echo "Unzipping $FILE_NAME..."
unzip "$FILE_NAME"

# Optional: Clean up
# rm "$FILE_NAME"

echo "Done."
