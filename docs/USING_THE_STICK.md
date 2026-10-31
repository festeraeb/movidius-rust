# Using the Movidius NCS1 as an independent second opinion

## What the old "Movidius validators" were meant to do

Two pieces of code in the Cesarops repo claimed to be a Movidius stick giving an independent second
opinion on a detection tile:

* `disciplines/fusion-detection/cesarops-detection/vision-workers/jitter_movidius.py` (Lock 3 jitter worker)
* `disciplines/fusion-detection/cesarops-detection/jitter-rs/src/validators/movidius.rs` (the `movidius`
  validator in `jitter-rs`)

The idea was sound: the primary pipeline (CPU `tract`, vision models) produces a candidate (material,
certainty, thermal evidence). A second, physically separate accelerator running its own network on the same
input votes on whether it agrees. Agreement raises certainty, disagreement lowers it. Heterogeneous hardware
and a separately trained network make correlated mistakes less likely.

The code never did that. `_thermal_heuristic` built its "result" from a hash of the tile id and the
coordinates; `independent_estimate` was a formula on the number of thermal bands. The stick was never asked
anything. Both were deleted; the evidence rows they produced are not to be trusted (they are listed in the
hand-off report, not changed here).

## What a real second opinion needs

1. A network trained for the question (for example thermal time-series or tile classifier) with honest
   held-out evaluation. Nothing on the fleet currently has this; the stick is only the runtime.
2. That network compiled to an NCSDK v1 graph (fp16).
3. This host code to load it and run tensors.
4. A caller that treats the vote as one input and records the raw output vector, the graph md5 and the
   device timing with the decision, so the vote can be audited.

Do not output a vote when the stick or graph is missing. Return "no vote".

## Rule: reset the stick after every graph, or it hangs

**After a graph has been loaded and run, reset the stick (`run --reset`, then `boot`) before you load any other graph.** Sending a second graph on top of a loaded one wedges the NCS1: USB writes time out, and the stick then drops off the bus (`device not accepting address ... error -71`, `unable to enumerate USB device`). This happened on 2026-10-29 when `gemm` ran after SqueezeNet without a reset.

- `run`, `gemm` and `serve` go through `v1::prepare_link`. Unless the same call just booted the stick from `03e7:2150`, it resets and re-boots it before uploading, and refuses to upload if the stick is not `waiting` afterwards. The state word alone is not enough: after a completed inference the stick reads `waiting` with the graph still loaded, and a second upload on top returned the first run's output with zero layer timings (seen 2026-10-30).
- Verified on hardware 2026-10-30: SqueezeNet twice in a row, then `gemm`, then SqueezeNet. Every load reset first, every SqueezeNet run was real (about 39 ms of device layer time, 44 C), and the stick never wedged.
- Open: the hand-built `gemm` graph makes the firmware reset itself about 3 s after upload, even on a freshly reset stick, and the tool then writes on its stale handle (`usb write: No such device`). The stick recovers on its own; the GEMM graph builder needs fixing.
- Once wedged, software cannot bring it back. `usbreset`, toggling the port's `disable` file and unbinding/rebinding the xHCI controller (0000:00:14.0) all left it un-enumerable, because none of them cut the port's power. Recovery is a power cut: unplug and replug the stick, or power-cycle the host. The BIOS may be able to switch USB power; a power cycle also restarts everything else on that machine. The daemon's auto-boot then brings the stick back.

## Using the stick now

### 1. Boot

The stick enumerates as `03e7:2150` (no firmware). Booting uploads Intel's firmware over USB bulk and it
re-enumerates as `03e7:f63b`.

```bash
movidius-rust boot            # default firmware/MvNCAPI-ma2450.mvcmd (see README for the exact file)
movidius-rust ping -c 20      # reads the device state word 20 times, prints measured round trips
```

`ping` output is real: each request is a 64-byte v1 header, each reply a 4-byte state word that must decode
to a known state (`waiting` is the idle state). Replies that do not decode are counted as invalid and are
not in the timing. Measured on this stick (earlier, on a USB 2.0 high-speed port; the stick now sits on a 5000M port and has not been re-pinged): min 72 us, median 79 us, mean 95 us.

The daemon (`movidius-daemon.service`, port 8185) auto-boots a stick that shows up as 2150 and answers
`GET /health` with a v1 status read.

### 2. Load a graph, run a tensor, read results

```bash
movidius-rust run -g squeezenet11.graph -i dog.jpg -l synset_words.txt -r 10 \
    --save-output out.f32 --save-input in.f32
```

What happens (all in `src/v1.rs`):

* `blobFile`: the graph file is sent; `auxBuffer`: a zeroed buffer is sent (the firmware fills it with
  per-layer timing, temperatures, throttling state and an error string).
* `config`: ten int32 (version, query, iterations, upper/lower temperature limit, three back-off times,
  temperature debug, network throttle) before the first tensor.
* `input1` / `input2` (alternating): the fp16 input tensor, HWC, channels last. Alternating the buffer name is
  required; reusing `input1` every time makes every second call return a stale result.
* `output`: fp16 result vector. Then `auxBuffer` is read back: device error string (120 bytes), 25 temperature
  samples in degrees C, thermal throttle level (0 none, 1 lower limit reached, 2 upper limit reached) and
  one float per layer with the milliseconds that layer took on the device.
* Deallocation is host-side only in this protocol. To load a different graph, reboot the stick
  (`run --reset`, then `boot`), as Intel's own library effectively does by resetting on close.

### 3. Check it against a CPU reference

`tools/ncs1-compile/compare_caffe_cpu.py` runs the same preprocessed input (saved with `--save-input`)
through Caffe fp32 on the CPU and compares top-5 and the output vector. Result for SqueezeNet v1.1 on
the same dog photo:

| | top-1 | top-2 | top-3 | top-4 | top-5 |
|---|---|---|---|---|---|
| CPU Caffe fp32 | 258 Samoyed 0.80884 | 259 Pomeranian 0.09747 | 248 Eskimo dog 0.02380 | 270 white wolf 0.01452 | 279 Arctic fox 0.01103 |
| Stick fp16 | 258 Samoyed 0.80566 | 259 Pomeranian 0.09778 | 248 Eskimo dog 0.02550 | 270 white wolf 0.01522 | 279 Arctic fox 0.01114 |

Same top-5 order. Max abs difference over the 1000 outputs: 0.003179 (class 258). Mean abs difference
0.0000077. Timing on the stick, 10 consecutive runs: wall 33.4 to 33.8 ms (median 33.6 ms, about 30
inferences per second), of which the firmware reports about 29.2 ms in the layers; the rest is USB transfer
(about 0.3 MB input in, 2 KB out). Graph upload was 7 ms. Temperature samples read about 52 to 54 C; the
throttle level stayed 0.

### 3b. Speed: where the time goes, and pipelining (measured 2026-10-29)

SqueezeNet v1.1 (`-s 12`), dog.jpg, 100 runs, stick on USB 3 (5000M), c4, daemon stopped, default firmware
options. `run -r 100 --pipeline [--no-aux]` runs the serial loop first and then the pipelined loop.

| mode | inferences/s | latency median | latency p90 |
|---|---|---|---|
| serial (submit, wait, read aux) | 29.08 | 34.28 ms | 34.47 ms |
| pipelined, two in flight, aux read | 29.40 | 68.68 ms | 69.01 ms |
| pipelined, two in flight, `--no-aux` (2 runs) | 29.50, 29.48 | 68.5 ms | 68.8 ms |

Pipelined latency is submit-to-result of each inference, so it includes waiting behind the one in front of it.

What this says:

* **The stick is the bottleneck, not USB or the host.** Keeping two inferences in flight (the firmware's
  `input1`/`input2` double buffer) hides the host side, and throughput rose only about 1.4 percent
  (29.1 to 29.5 per second). Each inference costs about 33.9 ms of device time back to back.
* The firmware's own layer timings sum to 29.2 ms. The other roughly 4.7 ms per inference stays when the host
  overlaps everything, so it is spent on the device (input conversion, DMA, result hand-off), not on the USB
  wire. Serial mode adds only about 0.4 ms of USB and host time on top.
* Per layer (ms, from `auxBuffer`): conv1 3.7, the final classifier convolution 4.2, and a handful of 1.5 to
  1.9 ms expand layers dominate. Nothing the host does changes those.
* Skipping the `auxBuffer` read (`--no-aux`) saves one USB round trip per inference and loses per-layer
  timing, temperature and the error string.
* Polling for the output now sleeps 150 us between tries and treats only a refused permit word as "not ready
  yet"; real USB errors surface immediately instead of being retried for 15 s.
* `--network-throttle 0` (a config word; Intel's host default is 1) made the stick slower, about 73 ms per
  inference. 2 was identical to 1. Leave it at 1.
* Correctness of the pipelined path: all 100 pipelined outputs were bit-identical (fp16 bit patterns) to the
  serial output, and the dog.jpg top-5 (Samoyed, Pomeranian, Eskimo dog, white wolf, Arctic fox) is unchanged.

So about 30 fps is what an NCS1 does for this net. More throughput needs a smaller network or more sticks.

Always `--reset` (or reboot the stick) between different graphs. Running `gemm` right after a `run` that
did not reset made the stick stop answering on USB until it was physically replugged (2026-10-29).

### HTTP backend (`serve`)

`movidius-rust serve --model X.graph --port N` serves one compiled graph: `GET /v1/models` (type
`vision_graph`, input shape, `generates_text: false`), `POST /v1/infer` (body = jpeg/png bytes, or raw
f32/f16 HWC with `?format=f32|f16`; query `top`, `output=1`, `aux=0`, `width`, `height`, `mean`, `scale`,
`order`) returning the top-k, wall ms and device ms. `/v1/chat/completions` and `/v1/completions` answer
501: the stick does not generate text. A commit earlier on 2026-10-29 returned a canned status string there
with invented token counts; that was removed.

### 4. Compile a graph

The v1 firmware only accepts "v1 graph" files (format version 2, produced by Intel's `mvNCCompile`). There is
no other source for them; the newer OpenVINO blob formats do not load on this firmware. The compiler is pure
Python and needs Caffe (or TensorFlow 1.x). `tools/ncs1-compile/README.md` has the exact recipe (toolkit
downloaded from the Internet Archive because Intel's mirror returns 403; compile in a no-network
Ubuntu 18.04 container). Rules we hit:

* The Caffe prototxt must have batch size 1 (`dim: 1`), otherwise the compiler stops with "more than one image
  inference not supported". The upstream SqueezeNet deploy file has batch 10.
* `-s 12` uses all 12 SHAVE cores.
* The compiler writes weights as fp16.

## What fits, and limits

* Precision: fp16 only. Inputs are converted to fp16, outputs are fp16 (about 3 decimal digits; see the
  0.003 max diff above). Do not use it where small probability differences matter.
* Batch size 1. One input tensor per call.
* Memory: the NCS1 has 512 MB of on-package RAM shared by firmware, graph and activations (Intel's published
  spec; not measured here). The host rejects graph files over 512 MiB and outputs over 64M values. The SqueezeNet
  graph is 2.4 MB.
* Networks: classic CNNs built from layers the v1 Caffe/TensorFlow parsers handle (convolution including
  depthwise, pooling, ReLU and variants, LRN, inner product, softmax, concat, eltwise, reshape and a few
  more; the authoritative list is `Controllers/CaffeParser.py` / `TensorFlowParser.py` in the toolkit).
  No recurrent or attention layers; custom layers are impossible because the firmware is closed. Check that
  a net compiles before planning around it. Only SqueezeNet v1.1 has been compiled and run here.
* Speed: about 30 fps for a 227x227 SqueezeNet. This is not faster than a modern CPU for the same net; its
  value is independence from the main compute and low power.
* Firmware black box: we boot Intel's `MvNCAPI.mvcmd`. We cannot inspect or change what runs on the SHAVE
  cores. Only the NCSDK 1.12.01.01 firmware (md5 0f0eb9f1...) was verified with this host; another build
  (md5 0ccf0926...) boots but drops off the bus and does not speak the v1 protocol.
* One process at a time: the USB interface can be claimed by one program. The daemon holds a lock while it
  talks to the stick; stop the daemon or use its endpoints when you run the CLI.
* Thermal: the firmware throttles at 85 and 95 C by default (lower and upper limits). Read the throttle
  level after every run if you depend on timing.

## Wiring it in as a validator (not built)

Make it a remote validator for `jitter-rs` (`JITTER_REMOTE_VALIDATORS=movidius_ncs1=http://c4:PORT`), where
the remote runs `movidius-rust` against a real compiled network, returns the raw output vector and graph md5,
and returns HTTP 503 when the stick or graph is not available so that `jitter-rs` skips the vote. There is
no trained network for the thermal question yet, so no validator was added.
