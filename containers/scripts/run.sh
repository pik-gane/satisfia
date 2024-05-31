#! /bin/sh
podman run --replace --name satisfia -v $PWD/scripts:/scripts -v $PWD/src:/src satisfia:latest "$@"
