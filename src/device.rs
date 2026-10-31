use anyhow::{Context, Result};
use rusb::{Device, DeviceDescriptor, DeviceHandle, GlobalContext};
use std::time::Duration;

pub const VID_MOVIDIUS: u16 = 0x03e7;

pub const PID_MYRIAD2_UNBOOTED: u16 = 0x2150;
pub const PID_MYRIAD2_BOOTED: u16 = 0xf63b;

pub const PID_MYRIADX_UNBOOTED: u16 = 0x2485;
pub const PID_MYRIADX_BOOTLOADER: u16 = 0xf63c;
pub const PID_MYRIADX_BOOTED: u16 = 0xf63d;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Unbooted,
    Booted,
    Unknown,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MovidiusDeviceInfo {
    pub bus_number: u8,
    pub address: u8,
    pub port_path: String,
    pub vid: u16,
    pub pid: u16,
    pub chip_generation: &'static str,
    pub state: DeviceState,
    pub manufacturer: Option<String>,
    pub product: Option<String>,
    pub serial_number: Option<String>,
}

#[allow(dead_code)]
impl MovidiusDeviceInfo {
    pub fn is_unbooted(&self) -> bool {
        self.state == DeviceState::Unbooted
    }
}

pub fn scan_devices() -> Result<Vec<MovidiusDeviceInfo>> {
    let mut found = Vec::new();
    let devices = rusb::devices().context("Failed to list USB devices")?;

    for device in devices.iter() {
        let desc = match device.device_descriptor() {
            Ok(d) => d,
            Err(_) => continue,
        };

        if desc.vendor_id() != VID_MOVIDIUS {
            continue;
        }

        let pid = desc.product_id();
        let (chip_gen, state) = match pid {
            PID_MYRIAD2_UNBOOTED => ("Myriad 2 (MA2450 / NCS1)", DeviceState::Unbooted),
            PID_MYRIAD2_BOOTED => ("Myriad 2 (MA2450 / NCS1)", DeviceState::Booted),
            PID_MYRIADX_UNBOOTED => ("Myriad X (MA2485 / NCS2)", DeviceState::Unbooted),
            PID_MYRIADX_BOOTLOADER => ("Myriad X (MA2485 / NCS2)", DeviceState::Unbooted),
            PID_MYRIADX_BOOTED => ("Myriad X (MA2485 / NCS2)", DeviceState::Booted),
            _ => ("Unknown Movidius VPU", DeviceState::Unknown),
        };

        let port_path = match device.port_numbers() {
            Ok(ports) => ports
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join("."),
            Err(_) => "unknown".to_string(),
        };

        let mut manufacturer = None;
        let mut product = None;
        let mut serial = None;

        if let Ok(handle) = device.open() {
            let timeout = Duration::from_millis(150);
            let languages = handle.read_languages(timeout).unwrap_or_default();
            if let Some(&lang) = languages.first() {
                if desc.manufacturer_string_index().is_some() {
                    manufacturer = handle.read_manufacturer_string(lang, &desc, timeout).ok();
                }
                if desc.product_string_index().is_some() {
                    product = handle.read_product_string(lang, &desc, timeout).ok();
                }
                if desc.serial_number_string_index().is_some() {
                    serial = handle.read_serial_number_string(lang, &desc, timeout).ok();
                }
            }
        }

        found.push(MovidiusDeviceInfo {
            bus_number: device.bus_number(),
            address: device.address(),
            port_path,
            vid: desc.vendor_id(),
            pid,
            chip_generation: chip_gen,
            state,
            manufacturer,
            product,
            serial_number: serial,
        });
    }

    Ok(found)
}

pub fn open_unbooted_device() -> Result<(Device<GlobalContext>, DeviceHandle<GlobalContext>, DeviceDescriptor)> {
    let devices = rusb::devices().context("Failed to list USB devices")?;

    for device in devices.iter() {
        let desc = match device.device_descriptor() {
            Ok(d) => d,
            Err(_) => continue,
        };

        if desc.vendor_id() == VID_MOVIDIUS && (desc.product_id() == PID_MYRIAD2_UNBOOTED || desc.product_id() == PID_MYRIADX_UNBOOTED) {
            let handle = device.open().with_context(|| {
                format!(
                    "Failed to open Movidius USB device (Bus {} Device {}). Check permissions/udev!",
                    device.bus_number(),
                    device.address()
                )
            })?;
            return Ok((device, handle, desc));
        }
    }

    anyhow::bail!("No unbooted Movidius device found on USB bus!")
}
