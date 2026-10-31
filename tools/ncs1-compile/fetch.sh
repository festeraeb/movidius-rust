#!/usr/bin/env bash
# Fetch the NCSDK v1.12.01.01 toolkit (compiler + matching firmware) into a NEW EMPTY directory.
# Usage: fetch.sh /path/to/new/empty/dir
# Intel's own mirror now returns 403 for this file; the Internet Archive snapshot still serves it.
set -euo pipefail
dest=${1:?usage: fetch.sh NEW_EMPTY_DIR}
mkdir -p "$dest"
[ -z "$(ls -A "$dest")" ] || { echo "$dest is not empty" >&2; exit 1; }
curl -fL -o "$dest/ncsdk.tar.gz" \
  "http://web.archive.org/web/20191119142208id_/https://downloadmirror.intel.com/28192/eng/NCSDK-1.12.01.01.tar.gz"
echo "df5261ab79697f3c564279010aebbc8fd171f57290a84f636201aee82368a3b2  $dest/ncsdk.tar.gz" | sha256sum -c -
mkdir "$dest/x"
tar xzf "$dest/ncsdk.tar.gz" -C "$dest/x" --no-same-owner --no-same-permissions
md5sum "$dest/x/NCSDK-1.12.01.01/ncsdk-x86_64/fw/MvNCAPI.mvcmd"   # expect 0f0eb9f1a529f2ef55c09d9fae72eb95
echo "toolkit at $dest/x/NCSDK-1.12.01.01/ncsdk-x86_64/tk ; firmware at .../fw/MvNCAPI.mvcmd"
