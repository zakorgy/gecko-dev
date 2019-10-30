

build:
	cargo build

all:
	cargo build
	cargo build --features panic
	cargo build --features panic,log
	cargo build --features panic,backtrace
	cargo build --features panic,log,backtrace
	cargo build --features log
	cargo build --features backtrace
	cargo build --features log,backtrace
	cargo build --no-default-features
	cargo build --no-default-features --features panic
	cargo build --no-default-features --features panic,log
	cargo build --no-default-features --features panic,backtrace
	cargo build --no-default-features --features panic,log,backtrace
	cargo build --no-default-features --features log
	cargo build --no-default-features --features backtrace
	cargo build --no-default-features --features log,backtrace
