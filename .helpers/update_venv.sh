#!/bin/bash

# We do this very carefully to make sure we change the occurences carefully

old_venv_name="$1"  # e.g. intro_scalable_dl_2022
new_venv_name="$2"  # e.g. intro_scalable_dl_2023

old_python_version="$3"  # e.g. 3.8.5
new_python_version="$4"  # e.g. 3.9.6

# List of regular expression to replace followed by the string to
# replace it with.
replacements=(
    "\"display_name\": \"$old_venv_name\""
    "\"display_name\": \"$new_venv_name\""

    "\"name\": \"$old_venv_name\""
    "\"name\": \"$new_venv_name\""

    "\"version\": \"$old_python_version\""
    "\"version\": \"$new_python_version\""
)

if [ "$(("${#replacements[@]}" % 2))" -ne 0 ]; then
    echo "Replacement array must have even length"
    exit 1
fi

for ((i=0; i < "${#replacements[@]}"; i+=2)); do
    old_string_regexp="${replacements[$i]}"
    new_string="${replacements["$((i + 1))"]}"
    find . -type f \
        | grep -v .git \
        | grep -v update_venv.sh \
        | xargs -d "\n" sed -i "s|$old_string_regexp|$new_string|g"
done
