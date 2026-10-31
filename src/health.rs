//! Real health checks: every number comes from a v1 status read whose 4-byte reply is
//! validated against the known device state values. Nothing is estimated.

use anyhow::Result;
use std::time::Instant;

use crate::v1::{Link, MyriadState};

#[derive(Debug, Clone)]
pub struct PingStats {
    pub sent: usize,
    /// Replies that arrived and decoded to a known state word.
    pub valid: usize,
    /// Replies that arrived but were not a known state word, or transfers that failed.
    pub invalid: usize,
    pub min_us: f64,
    pub avg_us: f64,
    pub p50_us: f64,
    pub max_us: f64,
    pub last_state: Option<MyriadState>,
    pub last_error: Option<String>,
}

/// Issue `count` status reads on an open link and time each request/reply pair.
/// RTT statistics only include valid replies.
pub fn ping(link: &Link, count: usize) -> PingStats {
    let mut rtts: Vec<f64> = Vec::with_capacity(count);
    let mut invalid = 0;
    let mut last = None;
    let mut last_error = None;
    for _ in 0..count {
        let t = Instant::now();
        match link.status() {
            Ok(s) if s.is_known() => {
                rtts.push(t.elapsed().as_secs_f64() * 1e6);
                last = Some(s);
            }
            Ok(s) => {
                invalid += 1;
                last_error = Some(format!("unknown state word: {}", s.name()));
            }
            Err(e) => {
                invalid += 1;
                last_error = Some(format!("{e:#}"));
            }
        }
    }
    let valid = rtts.len();
    let mut sorted = rtts.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (min_us, max_us, p50_us, avg_us) = if valid == 0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (sorted[0], sorted[valid - 1], sorted[valid / 2], rtts.iter().sum::<f64>() / valid as f64)
    };
    PingStats { sent: count, valid, invalid, min_us, avg_us, p50_us, max_us, last_state: last, last_error }
}

#[derive(Debug)]
pub struct Health {
    pub state: MyriadState,
    pub status_read_us: f64,
}

/// One v1 status read: the definition of "the stick is alive and speaking the v1 protocol".
pub fn check(link: &Link) -> Result<Health> {
    let t = Instant::now();
    let state = link.status()?;
    let us = t.elapsed().as_secs_f64() * 1e6;
    if !state.is_known() {
        anyhow::bail!("device replied with unknown state word ({})", state.name());
    }
    Ok(Health { state, status_read_us: us })
}
