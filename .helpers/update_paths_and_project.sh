#!/bin/bash

# We do this very carefully to make sure we change the occurences carefully

old_project="$1"  # e.g. training2107
new_project="$2"  # e.g. training2207

# List of regular expression to replace followed by the string to
# replace it with.
replacements=(
    "/p/project/$old_project/course_environment"
    "/p/project/$new_project/software_environment"

    "/p/project/$old_project/datasets"
    "/p/project/$new_project/datasets"

    "/p/project/$old_project"
    "/p/project/$new_project"

    "-A[= ]$old_project"
    "-A $new_project"

    "--account[= ]$old_project"
    "--account=$new_project"

    "$old_project"
    "$new_project"
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
        | grep -v update_paths_and_project.sh \
        | xargs -d "\n" sed -i "s|$old_string_regexp|$new_string|g"
done
