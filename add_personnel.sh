#!/bin/bash

# Usage: ./add_personnel.sh /path/to/folder/with/12/images 209_Ali_Tavakol
# Assumes API is running at http://localhost:5001 (change API_URL if needed)
# Folder must contain exactly 12 image files (any format, e.g., jpg, png)
# Person string format: personnel_id_first_name_last_name (e.g., 209_Ali_Tavakol)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_path> <person_string>"
    exit 1
fi

FOLDER_PATH="$1"
PERSON_STRING="$2"
API_URL="http://localhost:5001/personnel/upload"  # Change this to your actual API URL if different

# Ensure folder path ends with a slash
FOLDER_PATH="${FOLDER_PATH%/}/"

# Parse person_string
IFS='_' read -r PERSONNEL_ID FIRST_NAME LAST_NAME <<< "$PERSON_STRING"
if [ -z "$PERSONNEL_ID" ] || [ -z "$FIRST_NAME" ] || [ -z "$LAST_NAME" ]; then
    echo "Invalid person string format. Expected: personnel_id_first_name_last_name"
    exit 1
fi

# Get list of files in the folder (non-recursive)
FILES=("$FOLDER_PATH"*)
if [ ${#FILES[@]} -ne 12 ]; then
    echo "Folder must contain exactly 12 image files, found ${#FILES[@]}."
    exit 1
fi

# Check if all are files (not directories)
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "All items in folder must be files."
        exit 1
    fi
done

# Create a temporary file for JSON payload
TEMP_JSON=$(mktemp)

# Prepare base64 images array
IMAGES=()
for file in "${FILES[@]}"; do
    # Detect MIME type based on extension (fallback to image/jpeg)
    EXT="${file##*.}"
    case "$EXT" in
        png) MIME="image/png" ;;
        gif) MIME="image/gif" ;;
        bmp) MIME="image/bmp" ;;
        *) MIME="image/jpeg" ;;
    esac
    BASE64=$(base64 -w 0 "$file")
    IMAGES+=("data:$MIME;base64,$BASE64")
done

# Build JSON payload using jq with a temporary file to avoid argument size limit
{
    echo "{"
    echo "  \"images\": ["
    for ((i=0; i<${#IMAGES[@]}; i++)); do
        echo "    ${IMAGES[$i]}" | jq -R .
        [ $i -lt $((${#IMAGES[@]}-1)) ] && echo ","
    done
    echo "  ],"
    echo "  \"personnel_id\": \"$PERSONNEL_ID\","
    echo "  \"first_name\": \"$FIRST_NAME\","
    echo "  \"last_name\": \"$LAST_NAME\""
    echo "}"
} > "$TEMP_JSON"

# Send POST request with curl
RESPONSE=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    --data-binary @"$TEMP_JSON")

# Clean up temporary file
rm "$TEMP_JSON"

# Output response
echo "API Response:"
echo "$RESPONSE"

# Check if response contains an error
if echo "$RESPONSE" | grep -q "detail"; then
    echo "Error detected in API response. Please check the folder path and ensure the API is running."
    exit 1
fi