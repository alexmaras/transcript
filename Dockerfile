FROM rust:1.72


WORKDIR /usr/src/transcript
COPY . .

RUN cargo install --path .

CMD ["transcript"]
