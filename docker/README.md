# Deal with docker 

## Option 1: use bash files

1. Build docker image

```console
./create_image.sh
```

2. Launch docker image

```console
./launch_docker.sh
```

## Option 2: docker compose

Dependencies: [docker-compose](https://docs.docker.com/compose/)

Use docker compose yaml to build docker image and to launch container.