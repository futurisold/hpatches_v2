#!/bin/bash

base_url="http://icvl.ee.ic.ac.uk/vbalnt/hpatches"
hpatches_url="$base_url/hpatches-release.tar.gz"

if [ $# -eq 0 ]; then
    echo "Usage: "
    echo "sh download.sh hpatches || downloads the patches dataset"
    exit 1
fi

if test -d ./data/hpatches-release
then
echo "The ./data/hpatches-release directory already exists."
exit 0
fi

echo "\n>> Please wait, downloading the HPatches patches dataset ~4.2G\n"
mkdir -p ./data
wget -O ./data/hpatches-release.tar.gz $hpatches_url
echo ">> Please wait, extracting the HPatches patches dataset ~4.2G"
tar -xzf ./data/hpatches-release.tar.gz -C ./data
rm ./data/hpatches-release.tar.gz

