use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MA2X_MAGIC: &[u8; 4] = b"MA2x";

#[derive(Debug, Clone)]
pub struct MvcmdHeader {
    pub magic: [u8; 4],
    pub stage_len: u32,
    pub filename: String,
    pub total_size: usize,
}

#[derive(Debug, Clone)]
pub struct MvcmdSection {
    pub elf_name: String,
    pub section_name: String,
    pub paddr: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum MvcmdCommand {
    Label { elf_name: String, section_name: String },
    Write32 { addr: u32, value: u32 },
    Load { elf_name: String, section_name: String, addr: u32, data: Vec<u8> },
    CallInit { elf_name: String, entry: u32 },
    Exec { elf_name: String, entry: u32, mode: u8 },
}

#[derive(Debug, Clone)]
pub struct MvcmdImage {
    pub header: MvcmdHeader,
    pub commands: Vec<MvcmdCommand>,
    pub sections: Vec<MvcmdSection>,
    pub init_entry: Option<u32>,
    pub exec_entry: Option<u32>,
}

pub fn parse_mvcmd<P: AsRef<Path>>(path: P) -> Result<(MvcmdHeader, Vec<u8>)> {
    let path = path.as_ref();
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open firmware file at {}", path.display()))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read firmware file at {}", path.display()))?;

    if buffer.len() < 32 {
        bail!("Firmware file too small: {} bytes", buffer.len());
    }

    if &buffer[0..4] != MA2X_MAGIC {
        bail!(
            "Invalid firmware magic: expected 'MA2x', got {:02x?}",
            &buffer[0..4]
        );
    }

    let stage_len = u32::from_le_bytes(buffer[4..8].try_into().unwrap());

    let name_end = buffer[8..]
        .iter()
        .position(|&b| b == 0)
        .map(|pos| 8 + pos)
        .unwrap_or(32.min(buffer.len()));
    let filename = String::from_utf8_lossy(&buffer[8..name_end]).to_string();

    let header = MvcmdHeader {
        magic: *MA2X_MAGIC,
        stage_len,
        filename,
        total_size: buffer.len(),
    };

    Ok((header, buffer))
}

/// Dissect a complete Movidius `.mvcmd` (`MA2x`) boot stream into its constituent
/// commands, merged physical memory sections, and LEON4 SPARC V8 entry points.
pub fn dissect_mvcmd<P: AsRef<Path>>(path: P) -> Result<MvcmdImage> {
    let (header, data) = parse_mvcmd(path)?;
    let mut pos = 4usize;
    let mut cur_elf = String::new();
    let mut cur_sec = String::new();
    let mut commands = Vec::new();
    let mut sections: Vec<MvcmdSection> = Vec::new();
    let mut init_entry = None;
    let mut exec_entry = None;

    while pos < data.len() {
        let op = data[pos];
        match op {
            0xa8 => {
                if pos + 4 > data.len() {
                    bail!("Truncated 0xa8 label command at offset {pos}");
                }
                let slen = (data[pos + 1] as usize)
                    | ((data[pos + 2] as usize) << 8)
                    | ((data[pos + 3] as usize) << 16);
                if slen < 1 || pos + 4 + slen - 1 > data.len() {
                    bail!("Invalid 0xa8 label length {slen} at offset {pos}");
                }
                let raw = &data[pos + 4..pos + 4 + slen - 1];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let s = String::from_utf8_lossy(&raw[..end]).to_string();
                pos += 4 + slen - 1;
                if s.ends_with(".elf") {
                    cur_elf = s;
                } else {
                    cur_sec = s;
                }
                commands.push(MvcmdCommand::Label {
                    elf_name: cur_elf.clone(),
                    section_name: cur_sec.clone(),
                });
            }
            0x9a => {
                if pos + 9 > data.len() {
                    bail!("Truncated 0x9a WRITE32 command at offset {pos}");
                }
                let addr = u32::from_le_bytes(data[pos + 1..pos + 5].try_into().unwrap());
                let value = u32::from_le_bytes(data[pos + 5..pos + 9].try_into().unwrap());
                commands.push(MvcmdCommand::Write32 { addr, value });
                pos += 9;
            }
            0x8a => {
                if pos + 7 > data.len() {
                    bail!("Truncated 0x8a LOAD command at offset {pos}");
                }
                let addr = u32::from_le_bytes(data[pos + 1..pos + 5].try_into().unwrap());
                let words = u16::from_le_bytes(data[pos + 5..pos + 7].try_into().unwrap()) as usize;
                let nbytes = words * 4;
                if pos + 7 + nbytes > data.len() {
                    bail!("Truncated 0x8a LOAD payload at offset {pos} (need {nbytes} bytes)");
                }
                let chunk = data[pos + 7..pos + 7 + nbytes].to_vec();
                pos += 7 + nbytes;

                if let Some(last) = sections.last_mut() {
                    if last.elf_name == cur_elf
                        && last.section_name == cur_sec
                        && last.paddr.wrapping_add(last.data.len() as u32) == addr
                    {
                        last.data.extend_from_slice(&chunk);
                    } else {
                        sections.push(MvcmdSection {
                            elf_name: cur_elf.clone(),
                            section_name: cur_sec.clone(),
                            paddr: addr,
                            data: chunk.clone(),
                        });
                    }
                } else {
                    sections.push(MvcmdSection {
                        elf_name: cur_elf.clone(),
                        section_name: cur_sec.clone(),
                        paddr: addr,
                        data: chunk.clone(),
                    });
                }

                commands.push(MvcmdCommand::Load {
                    elf_name: cur_elf.clone(),
                    section_name: cur_sec.clone(),
                    addr,
                    data: chunk,
                });
            }
            0xba => {
                if pos + 5 > data.len() {
                    bail!("Truncated 0xba CALL_INIT command at offset {pos}");
                }
                let entry = u32::from_le_bytes(data[pos + 1..pos + 5].try_into().unwrap());
                init_entry = Some(entry);
                commands.push(MvcmdCommand::CallInit {
                    elf_name: cur_elf.clone(),
                    entry,
                });
                pos += 5;
            }
            0xa4 => {
                if pos + 6 > data.len() {
                    bail!("Truncated 0xa4 EXEC command at offset {pos}");
                }
                let entry = u32::from_le_bytes(data[pos + 1..pos + 5].try_into().unwrap());
                let mode = data[pos + 5];
                exec_entry = Some(entry);
                commands.push(MvcmdCommand::Exec {
                    elf_name: cur_elf.clone(),
                    entry,
                    mode,
                });
                pos += 6;
            }
            0x00 => {
                pos += 1;
            }
            other => {
                bail!("Unknown MA2x opcode 0x{other:02x} at byte offset {pos}");
            }
        }
    }

    Ok(MvcmdImage {
        header,
        commands,
        sections,
        init_entry,
        exec_entry,
    })
}

/// Build a standard 32-bit Little-Endian SPARC V8 (`EM_SPARC = 2`, `ELFDATA2LSB = 1`) ELF
/// from a subset of `MvcmdSection`s so tools like Ghidra, `rea`, and `llvm-objdump --triple=sparcel`
/// can inspect, disassemble, and decompile the firmware directly.
pub fn build_sparc_elf(sections: &[MvcmdSection], entry: u32) -> Vec<u8> {
    const EHDR_SIZE: usize = 52;
    const PHDR_SIZE: usize = 32;
    const SHDR_SIZE: usize = 40;

    let phnum = sections.len();
    let shnum = sections.len() + 2; // NULL + sections + .shstrtab

    let mut shstrtab = vec![0u8];
    let mut name_offsets = Vec::with_capacity(sections.len());
    for s in sections {
        name_offsets.push(shstrtab.len() as u32);
        shstrtab.extend_from_slice(s.section_name.as_bytes());
        shstrtab.push(0);
    }
    let shstrtab_name_off = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".shstrtab\0");

    let phoff = EHDR_SIZE;
    let shoff = phoff + phnum * PHDR_SIZE;
    let shstrtab_off = shoff + shnum * SHDR_SIZE;
    let mut cur_file_off = (shstrtab_off + shstrtab.len() + 15) & !15;

    let mut sec_file_offs = Vec::with_capacity(sections.len());
    for s in sections {
        sec_file_offs.push(cur_file_off);
        cur_file_off = (cur_file_off + s.data.len() + 15) & !15;
    }

    let mut out = vec![0u8; cur_file_off];

    // ELF32 Header (Little-Endian SPARC V8)
    out[0..4].copy_from_slice(b"\x7fELF");
    out[4] = 1; // ELFCLASS32
    out[5] = 1; // ELFDATA2LSB (Little-Endian)
    out[6] = 1; // EV_CURRENT
    out[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    out[18..20].copy_from_slice(&2u16.to_le_bytes()); // EM_SPARC
    out[20..24].copy_from_slice(&1u32.to_le_bytes()); // EV_CURRENT
    out[24..28].copy_from_slice(&entry.to_le_bytes());
    out[28..32].copy_from_slice(&(phoff as u32).to_le_bytes());
    out[32..36].copy_from_slice(&(shoff as u32).to_le_bytes());
    out[36..40].copy_from_slice(&0u32.to_le_bytes()); // e_flags
    out[40..42].copy_from_slice(&(EHDR_SIZE as u16).to_le_bytes());
    out[42..44].copy_from_slice(&(PHDR_SIZE as u16).to_le_bytes());
    out[44..46].copy_from_slice(&(phnum as u16).to_le_bytes());
    out[46..48].copy_from_slice(&(SHDR_SIZE as u16).to_le_bytes());
    out[48..50].copy_from_slice(&(shnum as u16).to_le_bytes());
    out[50..52].copy_from_slice(&((shnum - 1) as u16).to_le_bytes());

    // Program Headers & Section Headers
    for (i, s) in sections.iter().enumerate() {
        let foff = sec_file_offs[i] as u32;
        let filesz = s.data.len() as u32;
        let is_exec = s.section_name.contains("text") || s.section_name == ".init" || s.section_name == ".fini";
        let is_write = s.section_name.contains("data") || s.section_name.contains("Descriptors");

        // Program Header (PT_LOAD = 1)
        let mut p_flags = 4u32; // PF_R
        if is_write {
            p_flags |= 2; // PF_W
        }
        if is_exec {
            p_flags |= 1; // PF_X
        }
        let ph = &mut out[phoff + i * PHDR_SIZE..phoff + (i + 1) * PHDR_SIZE];
        ph[0..4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
        ph[4..8].copy_from_slice(&foff.to_le_bytes());
        ph[8..12].copy_from_slice(&s.paddr.to_le_bytes());
        ph[12..16].copy_from_slice(&s.paddr.to_le_bytes());
        ph[16..20].copy_from_slice(&filesz.to_le_bytes());
        ph[20..24].copy_from_slice(&filesz.to_le_bytes());
        ph[24..28].copy_from_slice(&p_flags.to_le_bytes());
        ph[28..32].copy_from_slice(&16u32.to_le_bytes());

        // Section Header (SHT_PROGBITS = 1)
        let mut sh_flags = 2u32; // SHF_ALLOC
        if is_write {
            sh_flags |= 1; // SHF_WRITE
        }
        if is_exec {
            sh_flags |= 4; // SHF_EXECINSTR
        }
        let sh = &mut out[shoff + (i + 1) * SHDR_SIZE..shoff + (i + 2) * SHDR_SIZE];
        sh[0..4].copy_from_slice(&name_offsets[i].to_le_bytes());
        sh[4..8].copy_from_slice(&1u32.to_le_bytes()); // SHT_PROGBITS
        sh[8..12].copy_from_slice(&sh_flags.to_le_bytes());
        sh[12..16].copy_from_slice(&s.paddr.to_le_bytes());
        sh[16..20].copy_from_slice(&foff.to_le_bytes());
        sh[20..24].copy_from_slice(&filesz.to_le_bytes());
        sh[32..36].copy_from_slice(&4u32.to_le_bytes());

        // Payload
        out[sec_file_offs[i]..sec_file_offs[i] + s.data.len()].copy_from_slice(&s.data);
    }

    // .shstrtab section header
    let sh_str = &mut out[shoff + (shnum - 1) * SHDR_SIZE..shoff + shnum * SHDR_SIZE];
    sh_str[0..4].copy_from_slice(&shstrtab_name_off.to_le_bytes());
    sh_str[4..8].copy_from_slice(&3u32.to_le_bytes()); // SHT_STRTAB
    sh_str[16..20].copy_from_slice(&(shstrtab_off as u32).to_le_bytes());
    sh_str[20..24].copy_from_slice(&(shstrtab.len() as u32).to_le_bytes());
    sh_str[32..36].copy_from_slice(&1u32.to_le_bytes());

    out[shstrtab_off..shstrtab_off + shstrtab.len()].copy_from_slice(&shstrtab);
    out
}

/// Unpack a `.mvcmd` firmware file into standard 32-bit SPARC V8 ELF files in `out_dir`.
pub fn unpack_mvcmd_to_elfs<P: AsRef<Path>, Q: AsRef<Path>>(
    mvcmd_path: P,
    out_dir: Q,
) -> Result<(MvcmdImage, Vec<PathBuf>)> {
    let img = dissect_mvcmd(&mvcmd_path)?;
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("Failed to create output directory {}", out_dir.display()))?;

    let mut elf_names: Vec<String> = Vec::new();
    for s in &img.sections {
        if !elf_names.contains(&s.elf_name) {
            elf_names.push(s.elf_name.clone());
        }
    }

    let mut written = Vec::new();
    for elf_name in elf_names {
        let secs: Vec<MvcmdSection> = img
            .sections
            .iter()
            .filter(|s| s.elf_name == elf_name)
            .cloned()
            .collect();
        let entry = if elf_name.contains("ddrInit") {
            img.init_entry.unwrap_or(0x70180000)
        } else {
            img.exec_entry.unwrap_or(0x80180000)
        };
        let elf_bytes = build_sparc_elf(&secs, entry);
        let stem = elf_name.trim_end_matches(".elf");
        let out_path = out_dir.join(format!("{stem}.sparc.elf"));
        std::fs::write(&out_path, &elf_bytes)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
        written.push(out_path);
    }

    Ok((img, written))
}
