#!/bin/zsh
set -e
game_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
open "$game_dir/index.html"
