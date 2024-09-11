#!/bin/bash

# Check if both input and output directories are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Create symbolic links for corpus_emb.*.pkl files
for file in "$input_dir"/corpus_emb.*.pkl; do
    if [ -f "$file" ]; then
        basename=$(basename "$file")
        # Get the absolute path of the input file
        absolute_path=$(readlink -f "$file")
        # Create a relative path from the output directory to the input file
        relative_path=$(realpath --relative-to="$output_dir" "$absolute_path")
        ln -sf "$relative_path" "$output_dir/$basename"
        echo "Created symlink: $output_dir/$basename -> $relative_path"
    fi
done

echo "Symlinking complete."