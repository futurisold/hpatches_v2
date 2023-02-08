#!/bin/bash

base_url="http://icvl.ee.ic.ac.uk/vbalnt/hpatches"
hpatches_url="$base_url/hpatches-release.tar.gz"
hpatches_sequences="$base_url/hpatches-sequences-release.tar.gz"

if test -d ./data/hpatches-release && test -d ./data/hpatches-sequences-release
then
echo "The dataset already exists!"
exit 0
fi

mkdir -p ./data
echo ">> Downloading..."
wget -O ./data/hpatches-release.tar.gz $hpatches_url --no-check-certificate
wget -O ./data/hpatches-sequences-release.tar.gz $hpatches_sequences --no-check-certificate
echo ">> Extracting..."
tar -xzf ./data/hpatches-release.tar.gz -C ./data
tar -xzf ./data/hpatches-sequences-release.tar.gz -C ./data
rm ./data/hpatches-release.tar.gz ./data/hpatches-sequences-release.tar.gz
echo ">> Done!"
