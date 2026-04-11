#!/bin/bash
# rename_strip_spaces.sh — Remove spaces from filenames in a directory

TARGET_DIR="${1:-.}"  # Default to current directory if no arg given

find "$TARGET_DIR" -maxdepth 1 -type f -name "* *" | while IFS= read -r filepath; do
    dir=$(dirname "$filepath")
    oldname=$(basename "$filepath")
    newname="${oldname// /_}"          # Replace spaces with underscores
    # newname="${oldname// /}"         # Uncomment to fully remove spaces instead

    echo "Renaming: '$oldname'  →  '$newname'"
    mv -- "$filepath" "$dir/$newname"
done
