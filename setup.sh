#! /bin/bash

# Install yt-dlp and jq first.
# brew install yt-dlp
# brew install jq

mkdir videos

# Download all the youtube videos
export LC_ALL=en_US.UTF-8
cat $1 |
while read in; do
    [[ $in =~ ^#.* ]] && continue
    id=$(yt-dlp --no-warnings --dump-single-json $in | jq -r '.id')
    title=$(yt-dlp --no-warnings --dump-single-json $in | jq -r '.title')
    path=./videos/$id.mp4
    if test -f "$path"; then
        echo "$title already downloaded."
    else
        echo "Downloading $title."
        yt-dlp --output $path $in
    fi
done

# Generate datapoints for frame to second mathching.

mkdir data

# So far only generates frames:
/usr/bin/python3 generate_data.py