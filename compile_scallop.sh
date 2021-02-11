#!/usr/bin/env bash
cd VQAR_all
cd Scallop
cargo build --release
cp ./target/release/libscallop.so ../VQAR/query_process/scallop.so