#!/bin/bash

# We do this very carefully to make sure we change the occurences carefully

old_partition="$1"  # e.g. booster
new_partition="$2"  # e.g. dc-gpu

# List of regular expression to replace followed by the string to
# replace it with.
replacements=(
    "-p[= ]$old_partition"
    "-p $new_partition"

    "--partition[= ]$old_partition"
    "--partition=$new_partition"
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
        | grep -v update_machine_and_partition.sh \
        | xargs -d "\n" sed -i "s|$old_string_regexp|$new_string|g"
done
