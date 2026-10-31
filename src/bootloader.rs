use anyhow::{bail, Context, Result};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::device::{open_unbooted_device, scan_devices, DeviceState};
use crate::mvcmd::{parse_mvcmd, MvcmdHeader};

pub const EP_BOOT_OUT: u8 = 0x01;
pub const INTERFACE_NUMBER: u8 = 0;
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024; // 64 KB

pub struct BootResult {
    pub header: MvcmdHeader,
    pub bytes_sent: usize,
    pub upload_duration: Duration,
    pub total_boot_duration: Duration,
    pub re_enumerated_pid: u16,
}

pub fn boot_device(firmware_path: &str) -> Result<BootResult> {
    let t_start = Instant::now();
    
    info!("Parsing Movidius firmware: {}", firmware_path);
    let (header, fw_bytes) = parse_mvcmd(firmware_path)
        .with_context(|| format!("Failed to parse firmware container: {}", firmware_path))?;
    info!(
        "Firmware validated: Magic={:?}, Target={}, Size={} bytes, StageLen={}",
        std::str::from_utf8(&header.magic).unwrap_or("????"),
        header.filename,
        header.total_size,
        header.stage_len
    );

    let (_device, handle, desc) = open_unbooted_device()?;
    info!(
        "Acquired unbooted device: VID: 0x{:04x}, PID: 0x{:04x}",
        desc.vendor_id(),
        desc.product_id()
    );

    // Detach kernel driver if active
    if let Ok(active) = handle.kernel_driver_active(INTERFACE_NUMBER) {
        if active {
            info!("Detaching active kernel driver on interface {}", INTERFACE_NUMBER);
            let _ = handle.detach_kernel_driver(INTERFACE_NUMBER);
        }
    }

    info!("Claiming interface {}", INTERFACE_NUMBER);
    handle.claim_interface(INTERFACE_NUMBER)
        .context("Failed to claim interface 0. Make sure udev rules or root permissions are set.")?;

    info!("Streaming firmware to endpoint 0x{:02x} in chunks...", EP_BOOT_OUT);
    let t_upload_start = Instant::now();
    let mut offset = 0;
    let total = fw_bytes.len();
    let chunk_size = DEFAULT_CHUNK_SIZE;
    let timeout = Duration::from_millis(3000);

    while offset < total {
        let end = (offset + chunk_size).min(total);
        let chunk = &fw_bytes[offset..end];

        match handle.write_bulk(EP_BOOT_OUT, chunk, timeout) {
            Ok(transferred) => {
                offset += transferred;
                if offset % (256 * 1024) == 0 || offset == total {
                    let pct = (offset as f64 / total as f64) * 100.0;
                    info!("Progress: {} / {} bytes ({:.1}%)", offset, total, pct);
                }
            }
            Err(e) => {
                // If device disconnects at the very end, it might be the ROM jumping to entry and resetting
                if offset > total * 90 / 100 {
                    warn!("Device dropped off USB during final packets ({:?}). This is expected as chip reboots.", e);
                    break;
                } else {
                    bail!("Bulk transfer failed at offset {}: {:?}", offset, e);
                }
            }
        }
    }

    let upload_duration = t_upload_start.elapsed();
    info!(
        "Firmware streaming completed: {} bytes uploaded in {:.2?}",
        offset,
        upload_duration
    );

    drop(handle);

    info!("Waiting for Movidius chip re-enumeration into active VSC mode...");
    let t_reboot_start = Instant::now();
    let max_wait = Duration::from_secs(10);
    let mut booted_pid = None;

    while t_reboot_start.elapsed() < max_wait {
        sleep(Duration::from_millis(300));
        let devices = scan_devices().unwrap_or_default();
        for dev in devices {
            if dev.state == DeviceState::Booted {
                booted_pid = Some(dev.pid);
                info!(
                    "Device re-enumerated successfully! VID: 0x{:04x}, PID: 0x{:04x} ({}) at Bus {} Addr {}",
                    dev.vid,
                    dev.pid,
                    dev.chip_generation,
                    dev.bus_number,
                    dev.address
                );
                break;
            }
        }
        if booted_pid.is_some() {
            break;
        }
    }

    let re_enumerated_pid = booted_pid.context(
        "Device did not re-enumerate as booted within 10 seconds. Check firmware compatibility!"
    )?;

    Ok(BootResult {
        header,
        bytes_sent: offset,
        upload_duration,
        total_boot_duration: t_start.elapsed(),
        re_enumerated_pid,
    })
}
