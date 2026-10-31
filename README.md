# movidius-rust

Pure-Rust host for the Intel Movidius Neural Compute Stick 1 (Myriad 2, MA2450): boot the stick over USB,
load a compiled network, run tensors, read results. No OpenVINO, no Intel NCSDK host library.
Intel's device firmware is a black box and is required.

See `docs/USING_THE_STICK.md` for the full guide, `NOTICE` for attribution (Apache-2.0 protocol sources).

## Build and use

```bash
cargo build --release
movidius-rust status                       # list sticks
movidius-rust boot                         # upload firmware (03e7:2150 -> 03e7:f63b)
movidius-rust ping -c 20                   # real v1 status reads, each reply validated
movidius-rust run -g net.graph -i img.jpg -l synset_words.txt -r 10
movidius-rust daemon                       # HTTP :8185 (see below), auto-boots a fresh stick
movidius-rust run -g net.graph -i img.jpg -r 100 --pipeline   # serial vs two-in-flight throughput
```

Daemon endpoints: `GET /health`, `GET /status`, `GET /v1/status`, `POST /ping?count=N`, `POST /boot`.
Every number reported comes from a v1 status read whose reply decoded to a known state word.

## Firmware (not committed)

Use the firmware that matches the v1 protocol and the graph compiler:

* Source: `https://downloadmirror.intel.com/28192/eng/NCSDK-1.12.01.01.tar.gz` (from `get_mvcmd.sh` in
  https://github.com/movidius/ncsdk). Intel's mirror now answers 403; the Internet Archive snapshot
  `http://web.archive.org/web/20191119142208id_/<same URL>` works (tarball sha256
  `df5261ab79697f3c564279010aebbc8fd171f57290a84f636201aee82368a3b2`).
* File inside: `NCSDK-1.12.01.01/ncsdk-x86_64/fw/MvNCAPI.mvcmd`, md5 `0f0eb9f1a529f2ef55c09d9fae72eb95`,
  save as `firmware/MvNCAPI-ma2450.mvcmd`. `tools/ncs1-compile/fetch.sh` does the download and checks.
* A different build, md5 `0ccf0926db369c23647a98b6468e9331` (1051388 bytes), boots to `03e7:f63b` but then
  drops off the bus and does not answer v1 traffic; do not use it.

## Measured (Intel NCS1 on a Linux x86-64 host, 2026-10-28/29)

| What | Result |
|---|---|
| SqueezeNet v1.1 on a real image | top-5 identical to Caffe fp32 on the CPU; largest output difference 0.0032 |
| Time per inference | about 33.9 ms, set by the stick: the firmware's own layer timings add up to 29.2 ms; USB and host take about 0.4 ms |
| Two inferences in flight | +1.4% throughput (29.1 to 29.5 per second), so host-side tuning is not worth chasing |

**One rule:** reset the stick after any graph has run and before loading another, or it can drop off the USB bus
until it is unplugged. The tool does this for you; see `docs/USING_THE_STICK.md`.

## With Lemonade

`movidius-rust serve` speaks enough of the OpenAI-style API for Lemonade to list it: `GET /v1/models` shows the
loaded vision graph (`"generates_text": false`), and real inference is `POST /v1/infer`. Chat and text completions
answer HTTP 501, because the NCS1 runs compiled vision graphs, not language models. It ships as part of
"Bitter Lemons" in the NautiK9 Lemonade Stand.

## Licence

Apache License 2.0 (`LICENSE`). The NCSDK v1 protocol work is credited in `NOTICE`. Intel's device firmware is
not included and is not covered by this licence; download it yourself as described above.
