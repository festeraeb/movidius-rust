//! Host side of the Movidius NCSDK v1 ("VSC" / usblink) protocol, written from scratch in Rust.
//!
//! This is an interoperability reimplementation of the wire protocol used by Intel's NCSDK v1
//! host library (api/src/usb_link_vsc.c and api/src/mvnc_api.c, Apache-2.0, tag v1.12.01.01,
//! https://github.com/movidius/ncsdk). No code is copied; only the on-the-wire behaviour
//! (64-byte header, 0xABCD permit word, buffer names, graph file offsets) is reproduced.
//! The device firmware (MvNCAPI.mvcmd) is Intel's and is treated as a black box.

use anyhow::{bail, Context, Result};
use rusb::{DeviceHandle, GlobalContext};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::device::{PID_MYRIAD2_BOOTED, VID_MOVIDIUS};

const EP_OUT: u8 = 0x01;
const EP_IN: u8 = 0x81;
const USB_TIMEOUT: Duration = Duration::from_millis(10_000);
const CHUNK: usize = 10 * 1024 * 1024;
const NAME_LEN: usize = 52;
const HEADER_LEN: usize = 64;
const PERMIT: u32 = 0xABCD;

// Host command codes (hostcommands_t)
const CMD_GET_STATUS: u8 = 0;
const CMD_RESET: u8 = 1;
const CMD_SET_DATA: u8 = 2;
const CMD_GET_DATA: u8 = 3;

// Graph (blob) file layout
pub const GRAPH_HEADER_LEN: usize = 264;
pub const GRAPH_STAGE_LEN: usize = 227;
const GRAPH_VERSION_OFFSET: usize = 36;
const GRAPH_VERSION: u8 = 2;
const GRAPH_NSTAGES_OFFSET: usize = 240;
const GRAPH_NOUT_OFFSET: usize = GRAPH_HEADER_LEN + 136;
const GRAPH_XSTRIDE_OFFSET: usize = GRAPH_HEADER_LEN + 172;

// auxBuffer layout returned after each inference
pub const DEBUG_LEN: usize = 120;
pub const THERMAL_LEN: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MyriadState {
    NotInit,
    Initialized,
    Waiting,
    Running,
    Finished,
    Pending,
    Other(u32),
}

impl MyriadState {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::NotInit,
            0x11 => Self::Initialized,
            0x22 => Self::Waiting,
            0x33 => Self::Running,
            0x44 => Self::Finished,
            0x55 => Self::Pending,
            o => Self::Other(o),
        }
    }
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Other(_))
    }
    pub fn name(&self) -> String {
        match self {
            Self::NotInit => "not_init".into(),
            Self::Initialized => "initialized".into(),
            Self::Waiting => "waiting".into(),
            Self::Running => "running".into(),
            Self::Finished => "finished".into(),
            Self::Pending => "pending".into(),
            Self::Other(v) => format!("unknown_0x{v:x}"),
        }
    }
}

/// Tunables sent to the device in the "config" buffer before the first tensor.
#[derive(Debug, Clone)]
pub struct Options {
    pub iterations: i32,
    pub temp_lim_upper: i32,
    pub temp_lim_lower: i32,
    pub backoff_normal_ms: i32,
    pub backoff_high_ms: i32,
    pub backoff_critical_ms: i32,
    pub temperature_debug: i32,
    pub network_throttle: i32,
}

impl Default for Options {
    fn default() -> Self {
        // Same defaults Intel's host library used.
        Self {
            iterations: 1,
            temp_lim_upper: 95,
            temp_lim_lower: 85,
            backoff_normal_ms: 0,
            backoff_high_ms: 100,
            backoff_critical_ms: 10_000,
            temperature_debug: 0,
            network_throttle: 1,
        }
    }
}

pub struct Link {
    h: DeviceHandle<GlobalContext>,
}

impl Link {
    /// Open the first booted (03e7:f63b) stick and claim interface 0.
    pub fn open() -> Result<Self> {
        let devices = rusb::devices().context("cannot list USB devices")?;
        for d in devices.iter() {
            let Ok(desc) = d.device_descriptor() else { continue };
            if desc.vendor_id() == VID_MOVIDIUS && desc.product_id() == PID_MYRIAD2_BOOTED {
                let h = d.open().context("cannot open booted Movidius device (permissions?)")?;
                if h.kernel_driver_active(0).unwrap_or(false) {
                    let _ = h.detach_kernel_driver(0);
                }
                h.claim_interface(0)
                    .context("cannot claim interface 0 (another process holds the stick?)")?;
                return Ok(Self { h });
            }
        }
        bail!("no booted Movidius NCS1 (03e7:f63b) found; boot it first")
    }

    fn write_all(&self, mut data: &[u8]) -> Result<()> {
        while !data.is_empty() {
            let n = data.len().min(CHUNK);
            let w = self.h.write_bulk(EP_OUT, &data[..n], USB_TIMEOUT).context("usb write")?;
            data = &data[w..];
        }
        Ok(())
    }

    fn read_exact(&self, buf: &mut [u8]) -> Result<()> {
        let mut off = 0;
        while off < buf.len() {
            let n = (buf.len() - off).min(CHUNK);
            let r = self.h.read_bulk(EP_IN, &mut buf[off..off + n], USB_TIMEOUT).context("usb read")?;
            off += r;
        }
        Ok(())
    }

    fn header(cmd: u8, hostready: bool, len: u32, offset: u32, name: &str) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0] = cmd;
        b[1] = hostready as u8;
        b[4..8].copy_from_slice(&len.to_le_bytes());
        b[8..12].copy_from_slice(&offset.to_le_bytes());
        let nb = name.as_bytes();
        assert!(nb.len() < NAME_LEN);
        b[12..12 + nb.len()].copy_from_slice(nb);
        b
    }

    /// Read the 4-byte device state word.
    pub fn status(&self) -> Result<MyriadState> {
        self.write_all(&Self::header(CMD_GET_STATUS, false, 0, 0, ""))?;
        let mut w = [0u8; 4];
        self.read_exact(&mut w)?;
        Ok(MyriadState::from_u32(u32::from_le_bytes(w)))
    }

    pub fn set_data(&self, name: &str, data: &[u8], hostready: bool) -> Result<()> {
        self.write_all(&Self::header(CMD_SET_DATA, hostready, data.len() as u32, 0, name))?;
        let mut w = [0u8; 4];
        self.read_exact(&mut w)?;
        let permit = u32::from_le_bytes(w);
        if permit != PERMIT {
            bail!("device refused set_data '{name}': permit word 0x{permit:x}");
        }
        self.write_all(data)
    }

    pub fn get_data(&self, name: &str, len: usize, offset: u32, hostready: bool) -> Result<Vec<u8>> {
        self.write_all(&Self::header(CMD_GET_DATA, hostready, len as u32, offset, name))?;
        let mut w = [0u8; 4];
        self.read_exact(&mut w)?;
        let permit = u32::from_le_bytes(w);
        if permit != PERMIT {
            bail!("device refused get_data '{name}': permit word 0x{permit:x}");
        }
        let mut out = vec![0u8; len];
        self.read_exact(&mut out)?;
        Ok(out)
    }

    /// Like `get_data`, but a refused permit word (output not ready yet) is `Ok(None)`.
    /// USB errors are still errors.
    pub fn try_get_data(&self, name: &str, len: usize, offset: u32, hostready: bool) -> Result<Option<Vec<u8>>> {
        self.write_all(&Self::header(CMD_GET_DATA, hostready, len as u32, offset, name))?;
        let mut w = [0u8; 4];
        self.read_exact(&mut w)?;
        if u32::from_le_bytes(w) != PERMIT {
            return Ok(None);
        }
        let mut out = vec![0u8; len];
        self.read_exact(&mut out)?;
        Ok(Some(out))
    }

    /// Ask the stick to reset itself. It re-enumerates as 03e7:2150 and must be booted again.
    pub fn reset(&self) -> Result<()> {
        self.write_all(&Self::header(CMD_RESET, false, 0, 0, ""))
    }

    pub fn wait_state(&self, accept: &[MyriadState], timeout: Duration) -> Result<MyriadState> {
        let t0 = Instant::now();
        loop {
            let s = self.status()?;
            if accept.contains(&s) {
                return Ok(s);
            }
            if t0.elapsed() > timeout {
                bail!("timeout waiting for device state {:?}, last state {}", accept, s.name());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// List the optimisation names the firmware advertises (config query mode).
    pub fn optimisation_list(&self) -> Result<Vec<String>> {
        let mut cfg = [0i32; 10];
        cfg[0] = 1;
        cfg[1] = 1; // query
        self.set_data("config", &i32s(&cfg), true)?;
        self.wait_state(&[MyriadState::Waiting, MyriadState::Finished], Duration::from_secs(15))?;
        let raw = self.get_data("optimizationList", 40 * 50, 0, false)?;
        cfg[1] = 0;
        self.set_data("config", &i32s(&cfg), false)?;
        Ok(raw
            .chunks(50)
            .map(|c| {
                let end = c.iter().position(|&b| b == b'~' || b == 0).unwrap_or(c.len());
                String::from_utf8_lossy(&c[..end]).to_string()
            })
            .filter(|s| !s.is_empty())
            .collect())
    }

    /// Send a compiled graph (NCSDK v1 blob, format version 2) to the stick.
    pub fn allocate_graph(&self, blob: &[u8], opts: Options) -> Result<Graph<'_>> {
        let (info, upload) = self.upload_graph(blob)?;
        Ok(Graph { link: LinkRef::Borrowed(self), info, opts, started: false, input_idx: false, pending: 0, upload })
    }

    /// Same as `allocate_graph`, but the graph owns the link (for long-lived servers).
    pub fn allocate_graph_owned(self: Arc<Self>, blob: &[u8], opts: Options) -> Result<Graph<'static>> {
        let (info, upload) = self.upload_graph(blob)?;
        Ok(Graph { link: LinkRef::Owned(self), info, opts, started: false, input_idx: false, pending: 0, upload })
    }

    fn upload_graph(&self, blob: &[u8]) -> Result<(GraphInfo, Duration)> {
        let info = GraphInfo::parse(blob)?;
        self.wait_state(&[MyriadState::Waiting], Duration::from_secs(10))
            .context("device not ready to accept a graph (is a previous graph still loaded? re-boot the stick)")?;
        let t0 = Instant::now();
        self.set_data("blobFile", blob, false)?;
        let aux = vec![0u8; 224 + 4 * info.nstages];
        self.set_data("auxBuffer", &aux, false)?;
        Ok((info, t0.elapsed()))
    }
}

/// Whatever can report the device state and reset it; a trait so the decision logic is testable
/// without a stick.
pub trait DeviceOps {
    fn state(&mut self) -> Result<MyriadState>;
    /// Reset the stick and boot it again; afterwards `state()` should read `Waiting`.
    fn reset_and_reboot(&mut self) -> Result<()>;
}

/// What to do before uploading a graph, given the device state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prep {
    Ready,
    ResetFirst,
}

/// `booted_here` is true only when this process booted the stick from 03e7:2150 just now. A stick
/// found already booted may hold a graph even though it reads `waiting`: after an inference completes
/// the state word goes back to `waiting` with the graph still resident (seen on hardware 2026-10-30:
/// a second upload on top returned the first run's output with zero layer timings). So only a stick
/// we booted ourselves and that reads `waiting` is ready; everything else is reset first.
pub fn decide_prep(booted_here: bool, state: &Result<MyriadState>) -> Prep {
    match state {
        Ok(MyriadState::Waiting) if booted_here => Prep::Ready,
        _ => Prep::ResetFirst,
    }
}

/// Make sure the device is clean and `Waiting` before a graph upload. Unless this process just booted it,
/// or if it is not `waiting` (busy, finished, or the state read fails), reset and re-boot it first.
/// Returns true if a reset happened.
pub fn ensure_waiting<D: DeviceOps>(d: &mut D, booted_here: bool) -> Result<bool> {
    let st = d.state();
    if decide_prep(booted_here, &st) == Prep::Ready {
        return Ok(false);
    }
    match &st {
        Ok(MyriadState::Waiting) => tracing::info!("stick was already booted and may hold a graph: resetting and re-booting before loading"),
        Ok(s) => tracing::warn!("device state is '{}', not 'waiting': resetting and re-booting before loading a graph", s.name()),
        Err(e) => tracing::warn!("device state read failed ({e:#}): resetting and re-booting before loading a graph"),
    }
    d.reset_and_reboot()?;
    match d.state()? {
        MyriadState::Waiting => Ok(true),
        s => bail!("device still '{}' after reset and re-boot", s.name()),
    }
}

struct RealDevice<'a> {
    firmware: &'a str,
}

impl DeviceOps for RealDevice<'_> {
    fn state(&mut self) -> Result<MyriadState> {
        Link::open()?.status()
    }
    fn reset_and_reboot(&mut self) -> Result<()> {
        if let Ok(l) = Link::open() {
            let _ = l.reset();
            drop(l);
            std::thread::sleep(Duration::from_millis(1200));
        }
        crate::bootloader::boot_device(self.firmware)?;
        Ok(())
    }
}

/// Open the stick ready to take a graph: boots it if it is not booted, and resets and re-boots it if
/// it was already booted (a graph may be resident) or is not `waiting`.
pub fn prepare_link(firmware: &str) -> Result<Link> {
    let booted_here = Link::open().is_err();
    if booted_here {
        tracing::info!("stick not booted, booting with {firmware}");
        crate::bootloader::boot_device(firmware)?;
    }
    ensure_waiting(&mut RealDevice { firmware }, booted_here)?;
    Link::open()
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn rd32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub nstages: usize,
    pub noutputs: usize,
}

impl GraphInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() < GRAPH_HEADER_LEN + GRAPH_STAGE_LEN || blob.len() > 512 * 1024 * 1024 {
            bail!("graph file has an implausible size ({} bytes)", blob.len());
        }
        if blob[GRAPH_VERSION_OFFSET] != GRAPH_VERSION {
            bail!(
                "unsupported graph file version {} (NCSDK v1 firmware wants {})",
                blob[GRAPH_VERSION_OFFSET],
                GRAPH_VERSION
            );
        }
        let nstages = blob[GRAPH_NSTAGES_OFFSET] as usize | ((blob[GRAPH_NSTAGES_OFFSET + 1] as usize) << 8);
        if nstages == 0 {
            bail!("graph has zero stages");
        }
        let last = (nstages - 1) * GRAPH_STAGE_LEN;
        if blob.len() < GRAPH_XSTRIDE_OFFSET + last + 4 {
            bail!("graph file truncated: stage table runs past the end");
        }
        let noutputs = (rd32(blob, GRAPH_NOUT_OFFSET + last)
            .wrapping_mul(rd32(blob, GRAPH_NOUT_OFFSET + last + 4))
            .wrapping_mul(rd32(blob, GRAPH_XSTRIDE_OFFSET + last))
            / 2) as usize;
        if noutputs == 0 || noutputs > 64 * 1024 * 1024 {
            bail!("graph output size looks wrong ({noutputs} values)");
        }
        Ok(Self { nstages, noutputs })
    }
}

enum LinkRef<'a> {
    Borrowed(&'a Link),
    Owned(Arc<Link>),
}

pub struct Graph<'a> {
    link: LinkRef<'a>,
    pub info: GraphInfo,
    pub opts: Options,
    started: bool,
    /// The device double-buffers inputs; the host alternates the buffer names input1/input2.
    input_idx: bool,
    /// Inferences submitted but not yet collected (0..=MAX_IN_FLIGHT).
    pending: usize,
    pub upload: Duration,
}

/// The firmware has two input buffers (input1/input2), so at most two inferences can be in flight.
pub const MAX_IN_FLIGHT: usize = 2;
/// Poll interval while waiting for an output. A failed poll costs one ~80 us USB round trip.
const POLL_SLEEP: Duration = Duration::from_micros(150);

/// What a submitted inference needs to be collected.
#[derive(Debug, Clone, Copy)]
pub struct Ticket {
    pub submitted: Instant,
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Output tensor as raw fp16 bit patterns.
    pub output_f16: Vec<u16>,
    /// Per-layer time on the device, milliseconds, as reported by the firmware.
    /// Empty when the aux read was skipped.
    pub layer_ms: Vec<f32>,
    /// Thermal samples reported by the firmware (25 floats, degrees C).
    pub thermal: Vec<f32>,
    /// 0 = none, 1 = lower limit reached, 2 = upper limit reached.
    pub throttle_level: i32,
    /// Non-empty when the firmware reports an error string.
    pub device_error: Option<String>,
    /// Host-measured wall time from the start of load_tensor to the end of get_result.
    pub wall: Duration,
}

impl InferenceResult {
    pub fn output_f32(&self) -> Vec<f32> {
        self.output_f16.iter().map(|&h| f16_to_f32(h)).collect()
    }
    pub fn device_total_ms(&self) -> f32 {
        self.layer_ms.iter().sum()
    }
}

impl Graph<'_> {
    fn link(&self) -> &Link {
        match &self.link {
            LinkRef::Borrowed(l) => l,
            LinkRef::Owned(l) => l,
        }
    }

    /// Ask the device to reset (it must be booted again afterwards).
    pub fn reset_device(&self) -> Result<()> {
        self.link().reset()
    }

    pub fn in_flight(&self) -> usize {
        self.pending
    }

    /// Send one input tensor (fp16 bit patterns, HWC) without waiting for the result.
    /// At most `MAX_IN_FLIGHT` may be outstanding; results come back in submission order via `collect`.
    pub fn submit(&mut self, input_f16: &[u16]) -> Result<Ticket> {
        if self.pending >= MAX_IN_FLIGHT {
            bail!("{MAX_IN_FLIGHT} inferences already in flight; collect one first");
        }
        let submitted = Instant::now();
        if !self.started {
            let o = &self.opts;
            let cfg = [
                1,
                0,
                o.iterations,
                o.temp_lim_upper,
                o.temp_lim_lower,
                o.backoff_normal_ms,
                o.backoff_high_ms,
                o.backoff_critical_ms,
                o.temperature_debug,
                o.network_throttle,
            ];
            self.link().set_data("config", &i32s(&cfg), false)?;
            self.started = true;
        }
        let bytes: Vec<u8> = input_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        let name = if self.input_idx { "input2" } else { "input1" };
        self.link().set_data(name, &bytes, true)?;
        self.input_idx = !self.input_idx;
        self.pending += 1;
        Ok(Ticket { submitted })
    }

    /// Wait for the oldest submitted inference. `want_aux = false` skips the auxBuffer read
    /// (saves one USB round trip; no per-layer timing, temperature or error string).
    pub fn collect(&mut self, ticket: Ticket, want_aux: bool) -> Result<InferenceResult> {
        if self.pending == 0 {
            bail!("collect() with nothing in flight");
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        let n_out = 2 * self.info.noutputs;
        let out_bytes = loop {
            if let Some(b) = self.link().try_get_data("output", n_out, 0, false)? {
                break b;
            }
            if Instant::now() > deadline {
                bail!("timed out waiting for inference output");
            }
            std::thread::sleep(POLL_SLEEP);
        };
        self.pending -= 1;
        let output_f16 = out_bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if !want_aux {
            return Ok(InferenceResult {
                output_f16,
                layer_ms: Vec::new(),
                thermal: Vec::new(),
                throttle_level: 0,
                device_error: None,
                wall: ticket.submitted.elapsed(),
            });
        }
        let aux_len = DEBUG_LEN + THERMAL_LEN + 4 + 4 * self.info.nstages;
        let aux = self.link().get_data("auxBuffer", aux_len, 0, false)?;
        let wall = ticket.submitted.elapsed();
        let dbg = &aux[..DEBUG_LEN];
        let device_error = if dbg[0] != 0 {
            let end = dbg.iter().position(|&b| b == 0).unwrap_or(DEBUG_LEN);
            Some(String::from_utf8_lossy(&dbg[..end]).to_string())
        } else {
            None
        };
        let thermal = aux[DEBUG_LEN..DEBUG_LEN + THERMAL_LEN]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let throttle_level = i32::from_le_bytes(aux[DEBUG_LEN + THERMAL_LEN..DEBUG_LEN + THERMAL_LEN + 4].try_into().unwrap());
        let layer_ms = aux[DEBUG_LEN + THERMAL_LEN + 4..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(InferenceResult { output_f16, layer_ms, thermal, throttle_level, device_error, wall })
    }

    /// Run one inference (submit + collect, with the aux read).
    pub fn infer(&mut self, input_f16: &[u16]) -> Result<InferenceResult> {
        let t = self.submit(input_f16)?;
        self.collect(t, true)
    }

    /// Run `n` inferences on the same input keeping two in flight. Returns results in order.
    /// Each result's `wall` is submit-to-collect latency for that inference.
    pub fn infer_pipelined(&mut self, input_f16: &[u16], n: usize, want_aux: bool) -> Result<Vec<InferenceResult>> {
        let mut results = Vec::with_capacity(n);
        let mut tickets = std::collections::VecDeque::new();
        let mut submitted = 0;
        while results.len() < n {
            while submitted < n && tickets.len() < MAX_IN_FLIGHT {
                tickets.push_back(self.submit(input_f16)?);
                submitted += 1;
            }
            let t = tickets.pop_front().expect("ticket queued");
            results.push(self.collect(t, want_aux)?);
        }
        Ok(results)
    }

    /// Deallocation is host-side only in this protocol; the device keeps the blob until reset.
    pub fn deallocate(self) {}
}

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = sign | ((e as u16) << 10) | (man >> 13) as u16;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // may carry into exponent, which is correct
    }
    h
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_values() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 2.0, 65504.0, -65504.0, 6.1035156e-5, 5.9604645e-8] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "value {v}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(1e9), 0x7c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_all_bit_patterns_roundtrip() {
        for h in 0u16..=0xffff {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "pattern 0x{h:04x}");
        }
    }

    #[test]
    fn header_layout_matches_reference() {
        let h = Link::header(CMD_SET_DATA, true, 0x1234, 7, "blobFile");
        assert_eq!(h.len(), 64);
        assert_eq!(h[0], 2);
        assert_eq!(h[1], 1);
        assert_eq!(&h[4..8], &0x1234u32.to_le_bytes());
        assert_eq!(&h[8..12], &7u32.to_le_bytes());
        assert_eq!(&h[12..20], b"blobFile");
        assert_eq!(h[20], 0);
    }

    #[test]
    fn graph_parse_rejects_bad_version() {
        let mut b = vec![0u8; GRAPH_HEADER_LEN + GRAPH_STAGE_LEN + 64];
        b[GRAPH_VERSION_OFFSET] = 3;
        assert!(GraphInfo::parse(&b).is_err());
    }

    struct Mock {
        states: Vec<Result<MyriadState>>,
        resets: usize,
    }
    impl DeviceOps for Mock {
        fn state(&mut self) -> Result<MyriadState> {
            if self.states.is_empty() {
                bail!("no more states");
            }
            self.states.remove(0)
        }
        fn reset_and_reboot(&mut self) -> Result<()> {
            self.resets += 1;
            Ok(())
        }
    }

    #[test]
    fn freshly_booted_waiting_needs_no_reset() {
        let mut m = Mock { states: vec![Ok(MyriadState::Waiting)], resets: 0 };
        assert!(!ensure_waiting(&mut m, true).unwrap());
        assert_eq!(m.resets, 0);
    }

    #[test]
    fn already_booted_waiting_resets_first() {
        // After a completed inference the stick reads `waiting` with the graph still loaded.
        let mut m = Mock { states: vec![Ok(MyriadState::Waiting), Ok(MyriadState::Waiting)], resets: 0 };
        assert!(ensure_waiting(&mut m, false).unwrap());
        assert_eq!(m.resets, 1);
        assert_eq!(decide_prep(false, &Ok(MyriadState::Waiting)), Prep::ResetFirst);
    }

    #[test]
    fn other_states_reset_first() {
        for booted_here in [true, false] {
            for s in [MyriadState::Finished, MyriadState::Running, MyriadState::Pending, MyriadState::Initialized, MyriadState::Other(7)] {
                let mut m = Mock { states: vec![Ok(s), Ok(MyriadState::Waiting)], resets: 0 };
                assert!(ensure_waiting(&mut m, booted_here).unwrap(), "{s:?}");
                assert_eq!(m.resets, 1);
            }
        }
    }

    #[test]
    fn state_read_error_resets_and_still_failing_is_an_error() {
        let mut m = Mock { states: vec![Err(anyhow::anyhow!("usb")), Ok(MyriadState::Waiting)], resets: 0 };
        assert!(ensure_waiting(&mut m, true).unwrap());
        let mut m = Mock { states: vec![Ok(MyriadState::Finished), Ok(MyriadState::Finished)], resets: 0 };
        assert!(ensure_waiting(&mut m, true).is_err());
        assert_eq!(m.resets, 1);
        assert_eq!(decide_prep(true, &Ok(MyriadState::Waiting)), Prep::Ready);
    }

    #[test]
    fn state_names() {
        assert_eq!(MyriadState::from_u32(0x22), MyriadState::Waiting);
        assert!(!MyriadState::from_u32(0x99).is_known());
    }
}
