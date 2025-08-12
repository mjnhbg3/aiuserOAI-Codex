#!/bin/sh
case "$1" in
  Username*) echo "x-access-token" ;;
  Password*) echo "$GIT_TOKEN" ;;
  *) echo ;;
esac
