use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

mod bootloader;
mod device;
pub mod graph;
mod health;
pub mod mvcmd;
mod run;
mod server;
pub mod v1;

const DEFAULT_FW: &str = "firmware/MvNCAPI-ma2450.mvcmd";

#[derive(Parser)]
#[command(name = "movidius-rust")]
#[command(about = "Pure-Rust host, firmware ELF dissector, and direct 12-SHAVE graph/GEMM compiler for Intel Movidius NCS1 (Myriad 2)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan USB buses and show status of all attached Movidius VPUs
    Status,

    /// Boot an unbooted stick with a firmware container (.mvcmd)
    Boot {
        #[arg(short, long, default_value = DEFAULT_FW)]
        firmware: String,
    },

    /// Inspect header and sections of a Movidius firmware container (.mvcmd)
    Info {
        #[arg(short, long, default_value = DEFAULT_FW)]
        firmware: String,
    },

    /// Unpack a Movidius .mvcmd firmware container into standard 32-bit SPARC V8 (sparcel) ELF binaries for Ghidra / REA / llvm-objdump
    UnpackFw {
        #[arg(short, long, default_value = DEFAULT_FW)]
        firmware: PathBuf,
        #[arg(short, long, default_value = "firmware/elf")]
        out_dir: PathBuf,
    },

    /// Inspect and dissect a compiled NCSDK v1 .graph file
    InspectGraph {
        #[arg(short, long)]
        graph: PathBuf,
    },

    /// Compile a pure-Rust 12-SHAVE matrix multiplication (Y = X * W + b) into a .graph and execute it on the Movidius VPU
    Gemm {
        /// Rows M of input matrix X[M, K] and output Y[M, N]
        #[arg(short = 'm', long, default_value_t = 64)]
        m: u32,
        /// Inner dimension K of X[M, K] and W[K, N]
        #[arg(short = 'k', long, default_value_t = 128)]
        k: u32,
        /// Columns N of weight matrix W[K, N] and output Y[M, N]
        #[arg(short = 'n', long, default_value_t = 64)]
        n: u32,
        /// Number of SHAVE vector cores to use (1..=12)
        #[arg(short = 's', long, default_value_t = 12)]
        shaves: u16,
        /// Apply hardware fused ReLU post-op on the SHAVE cores
        #[arg(long)]
        relu: bool,
        /// Number of hardware inference iterations to time
        #[arg(short = 'r', long, default_value_t = 10)]
        repeat: usize,
        /// Optional path to save the compiled .graph file
        #[arg(long)]
        save_graph: Option<PathBuf>,
        /// Reset the stick after the run (required before loading a different graph)
        #[arg(long, default_value_t = true)]
        reset: bool,
    },

    /// Real health check: read the device state word over the v1 protocol and time each read
    Ping {
        #[arg(short, long, default_value_t = 20)]
        count: usize,
    },

    /// Print the optimisation names the firmware advertises
    Optimisations,

    /// Run a compiled NCSDK v1 graph on an input file and print the result
    Run {
        /// Compiled graph (.graph, NCSDK v1 blob, format version 2)
        #[arg(short, long)]
        graph: PathBuf,
        /// Input: image (png/jpeg), or .raw/.f32 (f32 HWC) or .f16 (fp16 HWC), already preprocessed
        #[arg(short, long)]
        input: PathBuf,
        /// Input width for image decoding
        #[arg(long, default_value_t = 227)]
        width: u32,
        /// Input height for image decoding
        #[arg(long, default_value_t = 227)]
        height: u32,
        /// Per-channel mean, in network channel order (e.g. 104,117,123 for Caffe BGR)
        #[arg(long, default_value = "104,117,123")]
        mean: String,
        /// Pixel scale applied before mean subtraction
        #[arg(long, default_value_t = 1.0)]
        scale: f32,
        /// Channel order for image input: bgr or rgb
        #[arg(long, default_value = "bgr")]
        order: String,
        /// Label file, one label per line
        #[arg(short, long)]
        labels: Option<PathBuf>,
        /// Number of top classes to print
        #[arg(short = 'k', long, default_value_t = 5)]
        top: usize,
        /// Repeat the inference N times (timing statistics)
        #[arg(short, long, default_value_t = 1)]
        repeat: usize,
        /// Write the output vector (f32 little-endian) to this file
        #[arg(long)]
        save_output: Option<PathBuf>,
        /// Write the preprocessed input (f32 HWC little-endian) to this file, for CPU comparison
        #[arg(long)]
        save_input: Option<PathBuf>,
        /// After the serial runs, run the same number again with two inferences in flight
        /// (input1/input2) and report throughput and latency next to the serial numbers
        #[arg(long)]
        pipeline: bool,
        /// Firmware network_throttle config word (Intel host default 1; experimental)
        #[arg(long, default_value_t = 1)]
        network_throttle: i32,
        /// Skip the auxBuffer read in the pipelined runs (no per-layer timing, saves one USB round trip each)
        #[arg(long)]
        no_aux: bool,
        /// Reset the stick after the run (it must then be booted again)
        #[arg(long)]
        reset: bool,
    },

    /// Run background HTTP/REST API server and auto-boot watchdog daemon (:8185)
    Daemon {
        #[arg(short, long, default_value = "0.0.0.0:8185")]
        bind: String,
        #[arg(short, long, default_value = DEFAULT_FW)]
        firmware: String,
        #[arg(long, default_value_t = true)]
        auto_boot: bool,
    },

    /// Vision-graph HTTP backend for one compiled graph: /v1/models, POST /v1/infer (chat/completions answer 501)
    Serve {
        /// Path to the NCSDK v1 .graph checkpoint
        #[arg(long)]
        model: String,
        /// Bind host
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind port
        #[arg(long)]
        port: u16,
        /// Device index
        #[arg(long, default_value = "0")]
        device: String,
        /// Firmware used to (re)boot the stick when it is not booted or after a device error
        #[arg(long, default_value = DEFAULT_FW)]
        firmware: String,
    },
}

/// "min X median Y p90 Z max W" for a list of millisecond samples.
fn stats_line(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = s[((s.len() * 9) / 10).min(s.len() - 1)];
    format!("min {:.2} median {:.2} p90 {:.2} max {:.2}", s[0], s[s.len() / 2], p90, s[s.len() - 1])
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    match Cli::parse().command {
        Commands::Status => {
            let devices = device::scan_devices()?;
            if devices.is_empty() {
                println!("No Intel Movidius devices detected on any USB bus.");
            }
            for (i, dev) in devices.iter().enumerate() {
                println!("[{}] {}", i + 1, dev.chip_generation);
                println!("    VID:PID  : 0x{:04x}:0x{:04x}", dev.vid, dev.pid);
                println!("    State    : {:?}", dev.state);
                println!("    Location : Bus {} Port {}", dev.bus_number, dev.port_path);
            }
        }
        Commands::Boot { firmware } => match bootloader::boot_device(&firmware) {
            Ok(res) => {
                println!(
                    "Booted {} ({} bytes, upload {:.2?}, total {:.2?}), now PID 0x{:04x}",
                    res.header.filename,
                    res.bytes_sent,
                    res.upload_duration,
                    res.total_boot_duration,
                    res.re_enumerated_pid
                );
            }
            Err(e) => {
                error!("Boot failed: {e:#}");
                std::process::exit(1);
            }
        },
        Commands::Info { firmware } => {
            let img = mvcmd::dissect_mvcmd(&firmware)?;
            println!(
                "File: {}\nMagic: {:?}\nEmbedded name: {}\nStage length: {}\nTotal size: {} bytes",
                firmware,
                std::str::from_utf8(&img.header.magic).unwrap_or("?"),
                img.header.filename,
                img.header.stage_len,
                img.header.total_size
            );
            if let Some(init) = img.init_entry {
                println!("DDR Init Entry (LEON): 0x{init:08x}");
            }
            if let Some(exec) = img.exec_entry {
                println!("Main Exec Entry (LEON): 0x{exec:08x}");
            }
            println!("Physical Memory Sections ({}):", img.sections.len());
            for s in &img.sections {
                println!(
                    "  [{:<26}] {:<44} 0x{:08x}..0x{:08x} ({:>7} bytes)",
                    s.elf_name,
                    s.section_name,
                    s.paddr,
                    s.paddr.wrapping_add(s.data.len() as u32),
                    s.data.len()
                );
            }
        }
        Commands::UnpackFw { firmware, out_dir } => {
            let (img, paths) = mvcmd::unpack_mvcmd_to_elfs(&firmware, &out_dir)?;
            println!(
                "Dissected {} ({} bytes, {} sections) into {} SPARC V8 (sparcel) ELF file(s):",
                firmware.display(),
                img.header.total_size,
                img.sections.len(),
                paths.len()
            );
            for p in paths {
                println!("  -> {}", p.display());
            }
        }
        Commands::InspectGraph { graph } => {
            let blob = std::fs::read(&graph).with_context(|| format!("read {}", graph.display()))?;
            let dg = graph::dissect_graph(&blob)?;
            println!(
                "Graph: {}\nVersion: {}\nStages: {}\nSHAVEs: {}..={}\nBinary weights offset: {} ({} bytes of weights)",
                graph.display(),
                dg.version,
                dg.nstages,
                dg.first_shave,
                dg.last_shave,
                dg.binary_data_offset,
                dg.weights_len
            );
            for (i, s) in dg.stages.iter().enumerate() {
                println!(
                    "  [{i:2}] {:<22} op={:<2} k={}x{} s={}x{} p={}x{} in=({}x{}x{}) out=({}x{}x{}) in_ref=({},#{}) out_ref=({},#{}) post_op={}",
                    s.name,
                    s.op,
                    s.radix_y,
                    s.radix_x,
                    s.stride_y,
                    s.stride_x,
                    s.pad_y,
                    s.pad_x,
                    s.dim_x,
                    s.dim_y,
                    s.dim_z,
                    s.out_dim_x,
                    s.out_dim_y,
                    s.out_dim_z,
                    s.input.offset,
                    s.input.index,
                    s.output.offset,
                    s.output.index,
                    s.post_op
                );
            }
        }
        Commands::Gemm {
            m,
            k,
            n,
            shaves,
            relu,
            repeat,
            save_graph,
            reset,
        } => {
            // Deterministic non-trivial test matrices X[M, K], W[K, N], b[N]
            let mut x_f16 = Vec::with_capacity((m * k) as usize);
            for i in 0..(m * k) {
                let v = (((i as i32 * 17 + 5) % 41) - 20) as f32 * 0.05;
                x_f16.push(v1::f32_to_f16(v));
            }
            let mut w_f16 = Vec::with_capacity((k * n) as usize);
            for i in 0..(k * n) {
                let v = (((i as i32 * 31 + 11) % 37) - 18) as f32 * 0.025;
                w_f16.push(v1::f32_to_f16(v));
            }
            let mut b_f16 = Vec::with_capacity(n as usize);
            for i in 0..n {
                let v = ((i as i32 % 13) - 6) as f32 * 0.03;
                b_f16.push(v1::f32_to_f16(v));
            }

            let mut gb = graph::GraphBuilder::new(shaves);
            // The firmware expects the input-descriptor stage first, as every mvNCCompile graph has.
            let (dim_x, dim_y) = graph::GraphBuilder::factor_spatial_dims(m);
            gb.add_input_data_stage(dim_x, dim_y, k);
            gb.add_gemm_stage(
                "shave_gemm",
                m,
                k,
                n,
                &w_f16,
                Some(&b_f16),
                if relu { graph::PostOp::Relu } else { graph::PostOp::Bias },
                graph::BufferRef {
                    offset: 0,
                    index: graph::BUFFER_INPUT,
                },
                graph::BufferRef {
                    offset: 0,
                    index: graph::BUFFER_OUTPUT,
                },
            )?;
            let blob = gb.build()?;
            if let Some(p) = &save_graph {
                std::fs::write(p, &blob)?;
                println!("Saved compiled pure-Rust GEMM .graph ({} bytes) to {}", blob.len(), p.display());
            }

            // Boots the stick if needed and resets it first if a graph is already loaded.
            let link = v1::prepare_link(DEFAULT_FW)?;
            // Reset the stick even when the graph hangs, so a failure never leaves it wedged.
            let outcome = (|| -> Result<()> {
                let mut g = link.allocate_graph(&blob, v1::Options::default())?;
                println!(
                    "Uploaded pure-Rust GEMM graph [M={m}, K={k}, N={n}, shaves={shaves}, relu={relu}] ({} bytes) in {:.2?}",
                    blob.len(),
                    g.upload
                );

                let mut walls = Vec::new();
                let mut dev_ms = Vec::new();
                let mut last = None;
                for i in 0..repeat.max(1) {
                    let r = g.infer(&x_f16)?;
                    if let Some(e) = &r.device_error {
                        anyhow::bail!("Movidius firmware reported error: {e}");
                    }
                    walls.push(r.wall.as_secs_f64() * 1e3);
                    dev_ms.push(r.device_total_ms() as f64);
                    println!(
                        "  run {:>2}: USB wall {:.2} ms | SHAVE hardware kernel {:.3} ms | temp {:.1} C | throttle {}",
                        i + 1,
                        r.wall.as_secs_f64() * 1e3,
                        r.device_total_ms(),
                        r.thermal.first().copied().unwrap_or(0.0),
                        r.throttle_level
                    );
                    last = Some(r);
                }
                let r = last.unwrap();
                let stick_out = r.output_f32();
                let cpu_ref = graph::cpu_gemm_f16_ref(
                    m as usize,
                    k as usize,
                    n as usize,
                    &x_f16,
                    &w_f16,
                    Some(&b_f16),
                    relu,
                );

                let mut max_err = 0.0f32;
                let mut sum_err = 0.0f64;
                for (&sv, &cv) in stick_out.iter().zip(cpu_ref.iter()) {
                    let diff = (sv - cv).abs();
                    if diff > max_err {
                        max_err = diff;
                    }
                    sum_err += diff as f64;
                }
                let mean_err = sum_err / (stick_out.len() as f64);
                dev_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med_dev_ms = dev_ms[dev_ms.len() / 2];
                let ops = 2.0 * (m as f64) * (k as f64) * (n as f64);
                let gflops = if med_dev_ms > 0.0 {
                    (ops / (med_dev_ms * 1e-3)) / 1e9
                } else {
                    0.0
                };

                println!("=== Verification vs CPU Reference ({} outputs) ===", stick_out.len());
                println!("  Stick first 5 : {:?}", &stick_out[..stick_out.len().min(5)]);
                println!("  CPU   first 5 : {:?}", &cpu_ref[..cpu_ref.len().min(5)]);
                println!("  Max abs error : {:.6} (FP16 machine epsilon ~0.001)", max_err);
                println!("  Mean abs error: {:.6}", mean_err);
                println!("  Median SHAVE  : {:.3} ms ({:.2} GFLOPS on {} SHAVEs)", med_dev_ms, gflops, shaves);

                g.deallocate();
                Ok(())
            })();
            if reset || outcome.is_err() {
                let _ = link.reset();
                std::thread::sleep(std::time::Duration::from_millis(1200));
                let _ = bootloader::boot_device(DEFAULT_FW);
            }
            outcome?;
        }
        Commands::Ping { count } => {
            let link = v1::Link::open()?;
            let s = health::ping(&link, count);
            println!("status reads sent     : {}", s.sent);
            println!("valid replies         : {}", s.valid);
            println!("invalid/failed        : {}", s.invalid);
            if s.valid > 0 {
                println!(
                    "RTT us min/p50/avg/max: {:.0} / {:.0} / {:.0} / {:.0}",
                    s.min_us, s.p50_us, s.avg_us, s.max_us
                );
            }
            if let Some(st) = s.last_state {
                println!("device state          : {}", st.name());
            }
            if let Some(e) = &s.last_error {
                println!("last error           : {e}");
            }
            if s.valid == 0 {
                std::process::exit(1);
            }
        }
        Commands::Optimisations => {
            let link = v1::Link::open()?;
            for o in link.optimisation_list()? {
                println!("{o}");
            }
        }
        Commands::Run {
            graph,
            input,
            width,
            height,
            mean,
            scale,
            order,
            labels,
            top,
            repeat,
            save_output,
            save_input,
            pipeline,
            network_throttle,
            no_aux,
            reset,
        } => {
            let m: Vec<f32> = mean
                .split(',')
                .map(|s| s.trim().parse::<f32>())
                .collect::<std::result::Result<_, _>>()
                .context("--mean needs 3 comma separated numbers")?;
            anyhow::ensure!(m.len() == 3, "--mean needs exactly 3 values");
            let pre = run::Preprocess {
                width,
                height,
                mean: [m[0], m[1], m[2]],
                scale,
                bgr: order.eq_ignore_ascii_case("bgr"),
            };
            let blob = std::fs::read(&graph).with_context(|| format!("read {}", graph.display()))?;
            let input_f16 = run::load_input(&input, &pre)?;
            if let Some(p) = save_input {
                let f: Vec<u8> = input_f16
                    .iter()
                    .flat_map(|&h| v1::f16_to_f32(h).to_le_bytes())
                    .collect();
                std::fs::write(p, f)?;
            }
            let labels = labels.map(|p| run::load_labels(&p)).transpose()?;
            let link = v1::prepare_link(DEFAULT_FW)?;
            // Run the device work in a closure so --reset still fires when it fails;
            // otherwise a hung graph leaves the stick wedged and needing a physical replug.
            let outcome = (|| -> Result<()> {
                let mut g = link.allocate_graph(&blob, v1::Options { network_throttle, ..Default::default() })?;
                info!(
                    "graph: {} stages, {} outputs, blob uploaded in {:.1?}",
                    g.info.nstages, g.info.noutputs, g.upload
                );
                let mut walls = Vec::new();
                let mut last = None;
                let serial_t0 = std::time::Instant::now();
                for i in 0..repeat.max(1) {
                    let r = g.infer(&input_f16)?;
                    if let Some(e) = &r.device_error {
                        anyhow::bail!("device reported an error: {e}");
                    }
                    println!(
                        "run {:>3}: wall {:.2} ms, device layers sum {:.2} ms, throttle level {}",
                        i + 1,
                        r.wall.as_secs_f64() * 1e3,
                        r.device_total_ms(),
                        r.throttle_level
                    );
                    walls.push(r.wall.as_secs_f64() * 1e3);
                    last = Some(r);
                }
                let serial_total = serial_t0.elapsed();
                let r = last.unwrap();
                let out = r.output_f32();
                for (rank, (idx, v)) in run::top_k(&out, top).into_iter().enumerate() {
                    let name = labels
                        .as_ref()
                        .and_then(|l| l.get(idx))
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    println!("top{}: class {:>4}  value {:.5}  {}", rank + 1, idx, v, name);
                }
                if walls.len() > 1 {
                    let mut s = walls.clone();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    println!(
                        "wall ms over {} runs: min {:.2} median {:.2} max {:.2}",
                        s.len(),
                        s[0],
                        s[s.len() / 2],
                        s[s.len() - 1]
                    );
                }
                if walls.len() > 1 {
                    println!(
                        "SERIAL    : {} runs in {:.1} ms = {:.2} inferences/s; latency ms {}",
                        walls.len(),
                        serial_total.as_secs_f64() * 1e3,
                        walls.len() as f64 / serial_total.as_secs_f64(),
                        stats_line(&walls)
                    );
                }
                if pipeline {
                    let n = repeat.max(2);
                    let t0 = std::time::Instant::now();
                    let res = g.infer_pipelined(&input_f16, n, !no_aux)?;
                    let total = t0.elapsed();
                    let lat: Vec<f64> = res.iter().map(|x| x.wall.as_secs_f64() * 1e3).collect();
                    println!(
                        "PIPELINED : {} runs in {:.1} ms = {:.2} inferences/s; latency ms {} (aux read {})",
                        n,
                        total.as_secs_f64() * 1e3,
                        n as f64 / total.as_secs_f64(),
                        stats_line(&lat),
                        if no_aux { "skipped" } else { "on" }
                    );
                    // Correctness: every pipelined output must match the serial output.
                    let serial_bits = &r.output_f16;
                    let mut identical = 0usize;
                    let mut max_diff = 0.0f32;
                    let mut top_same = true;
                    let serial_top: Vec<usize> = run::top_k(&out, top).iter().map(|x| x.0).collect();
                    for x in &res {
                        if &x.output_f16 == serial_bits {
                            identical += 1;
                        }
                        let o = x.output_f32();
                        for (a, b) in o.iter().zip(out.iter()) {
                            max_diff = max_diff.max((a - b).abs());
                        }
                        let t: Vec<usize> = run::top_k(&o, top).iter().map(|x| x.0).collect();
                        top_same &= t == serial_top;
                        if let Some(e) = &x.device_error {
                            anyhow::bail!("device reported an error during pipelined run: {e}");
                        }
                    }
                    println!(
                        "pipelined check: {identical}/{n} outputs bit-identical to serial, max abs diff {max_diff:.6}, top-{top} order identical: {top_same}"
                    );
                    if !top_same || max_diff > 0.01 {
                        anyhow::bail!("pipelined outputs differ from the serial output");
                    }
                }
                println!(
                    "per-layer device ms: {}",
                    r.layer_ms
                        .iter()
                        .map(|x| format!("{x:.2}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let temps: Vec<String> = r.thermal.iter().take(3).map(|x| format!("{x:.1}")).collect();
                println!("thermal samples (first 3): {}", temps.join(" "));
                if let Some(p) = save_output {
                    std::fs::write(p, out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                }
                g.deallocate();
                Ok(())
            })();
            if reset || outcome.is_err() {
                let _ = link.reset();
                info!("reset requested; stick will re-enumerate as 03e7:2150");
            }
            outcome?;
        }
        Commands::Daemon {
            bind,
            firmware,
            auto_boot,
        } => {
            server::start_server(&bind, firmware, auto_boot)?;
        }
        Commands::Serve {
            model,
            host,
            port,
            device,
            firmware,
        } => {
            server::start_lemonade_server(&model, &host, port, &device, &firmware)?;
        }
    }
    Ok(())
}
