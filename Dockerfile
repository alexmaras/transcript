FROM rust:1.72

RUN apt-get update && apt-get install -y clang cmake

WORKDIR /usr/src/transcript
COPY . .

RUN cargo install --path .

CMD ["transcript"]
