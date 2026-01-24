#!/bin/bash
# Update SPINE submodule to a specific version (tag or commit)
# Usage: ./update_spine_version.sh <version>
# Example: ./update_spine_version.sh v0.9.3

set -e

if [ -z "$1" ]; then
    echo "Error: No version specified"
    echo "Usage: $0 <version>"
    echo ""
    echo "Available tags:"
    cd spine && git fetch --tags && git tag -l --sort=-v:refname | head -10
    exit 1
fi

VERSION=$1

echo "Updating SPINE submodule to $VERSION..."

cd spine
git fetch --tags
git checkout "$VERSION"
cd ..

git add spine

echo ""
echo "SPINE submodule updated to $VERSION"
echo "Current submodule status:"
git submodule status
echo ""
echo "Commit this change with:"
echo "  git commit -m \"Update SPINE to $VERSION\""
