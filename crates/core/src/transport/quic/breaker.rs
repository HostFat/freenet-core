//! The QUIC crash-loop breaker (#5940).
//!
//! If a run with QUIC enabled ends abnormally, QUIC stays off for
//! [`QUIC_BREAKER_COOLDOWN`]. Some failures inside quinn's own code end the
//! process rather than a task, and the service manager restarts the node into
//! the same failure; the breaker bounds that to one restart with QUIC per
//! cooldown.
//!
//! The mechanism is a marker file in the node's data directory,
//! [`MARKER_FILE`]:
//!
//! - **Armed.** Written (atomically) when the node is about to enable QUIC.
//!   It records the process id.
//! - **Removed** on a clean exit: an operator stop (SIGTERM, Ctrl+C) or an
//!   auto-update exit. The `freenet` binary calls [`quic_breaker_clean_exit`]
//!   on exactly those paths, so an abort, a signal death, a forced exit from
//!   the library or an error exit all leave it in place.
//! - **Tripped.** At startup, an Armed marker left by an earlier process means
//!   that run ended abnormally. The marker is rewritten as Tripped with the
//!   time, and QUIC stays off. Later restarts inside the cooldown read the
//!   Tripped time and stay off without writing anything. After the cooldown
//!   the marker is armed again and QUIC is on.
//!
//! The run wrapper (`freenet service run-wrapper`, the tray on macOS and
//! Windows) stops its child with a hard kill for a tray Restart, Stop, Quit or
//! update. It writes the child's pid to the file named by
//! [`STOP_NOTE_ENV_VAR`] first ([`record_deliberate_stop`]), and an Armed
//! marker whose pid matches that note is treated as a clean exit.
//!
//! Nodes with QUIC off never read or write the marker. The breaker is off
//! unless the `freenet` binary enables it ([`enable_quic_crash_breaker`]), so
//! tests and embedders that build nodes in-process never see it.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// How long QUIC stays off after a run with QUIC enabled ended abnormally.
pub(crate) const QUIC_BREAKER_COOLDOWN: Duration = Duration::from_secs(24 * 60 * 60);

/// The marker's file name in the node's data directory.
pub(crate) const MARKER_FILE: &str = "quic-crash-breaker.json";

/// Environment variable the run wrapper sets on its child: the path of the
/// file it writes the child's pid to before stopping it with a hard kill.
pub const STOP_NOTE_ENV_VAR: &str = "FREENET_QUIC_BREAKER_STOP_NOTE";

/// Set by [`enable_quic_crash_breaker`]; off in tests and embedders.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// The marker this process armed, so a clean exit removes exactly that file.
static ARMED: OnceLock<PathBuf> = OnceLock::new();

/// Turn the breaker on for this process. Called once by the `freenet` binary
/// before the node is built, alongside the other process-level switches
/// (`enable_abort_on_fatal_listener_exit`).
pub fn enable_quic_crash_breaker() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Remove the marker this process armed. The `freenet` binary calls this on
/// its clean exits (an operator stop and an auto-update exit) and on no other.
/// A no-op when this process armed nothing.
pub fn quic_breaker_clean_exit() {
    if let Some(path) = ARMED.get() {
        if let Err(error) = std::fs::remove_file(path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    %error,
                    path = %path.display(),
                    "Could not remove the QUIC crash-breaker marker on a clean exit; \
                     the next start will run with QUIC off for the cooldown"
                );
            }
        }
    }
}

/// Record that the run wrapper is about to stop process `pid` on purpose, so
/// the breaker does not read the hard kill as an abnormal exit. Best effort.
pub fn record_deliberate_stop(note: &Path, pid: u32) {
    let _best_effort = write_atomic(note, pid.to_string().as_bytes());
}

/// The marker's contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum Marker {
    /// A process with QUIC enabled is running, or ended without a clean exit.
    Armed { pid: u32, armed_at_secs: u64 },
    /// The breaker tripped at this time (seconds since the Unix epoch).
    Tripped { tripped_at_secs: u64 },
}

/// What the breaker decided at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// QUIC is not configured on; nothing was read or written.
    NotRequested,
    /// The operator override is set; QUIC is on and the marker is untouched.
    Ignored,
    /// QUIC is on and the marker is armed.
    Armed,
    /// QUIC is on, but the marker could not be written.
    Unarmed,
    /// The previous run with QUIC enabled ended abnormally: the trip was
    /// recorded now, and QUIC is off until `until`.
    Tripped { until: SystemTime },
    /// QUIC is off until `until`, from an earlier trip.
    Cooling { until: SystemTime },
}

impl Decision {
    /// Whether QUIC runs.
    pub(crate) fn quic_enabled(self) -> bool {
        matches!(self, Self::Ignored | Self::Armed | Self::Unarmed)
    }
}

/// Inputs to [`decide`], so tests can set the time, pid and note path.
pub(crate) struct Startup<'a> {
    pub data_dir: &'a Path,
    pub now: SystemTime,
    pub pid: u32,
    pub ignore_breaker: bool,
    pub stop_note: Option<&'a Path>,
}

pub(crate) fn marker_path(data_dir: &Path) -> PathBuf {
    data_dir.join(MARKER_FILE)
}

/// Decide whether QUIC runs, reading and writing the marker as described in
/// the module docs. Logs nothing and touches no global state.
pub(crate) fn decide(quic_requested: bool, startup: &Startup<'_>) -> Decision {
    if !quic_requested {
        return Decision::NotRequested;
    }
    if startup.ignore_breaker {
        return Decision::Ignored;
    }
    let path = marker_path(startup.data_dir);
    let now_secs = unix_secs(startup.now);
    let deliberate_stop = startup.stop_note.and_then(take_stop_note);

    let previous = match std::fs::read(&path) {
        Ok(bytes) => Some(
            // A marker this binary cannot parse is read as unclean: it is
            // only ever written whole (atomic rename), so a damaged one is
            // not a normal state.
            serde_json::from_slice::<Marker>(&bytes).unwrap_or(Marker::Armed {
                pid: 0,
                armed_at_secs: 0,
            }),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => None,
    };

    match previous {
        Some(Marker::Armed { pid, .. }) if deliberate_stop != Some(pid) => {
            let tripped = Marker::Tripped {
                tripped_at_secs: now_secs,
            };
            // Off even if the write fails: the evidence is in hand now.
            let _best_effort = write_marker(&path, tripped);
            Decision::Tripped {
                until: cooldown_end(now_secs),
            }
        }
        Some(Marker::Tripped { tripped_at_secs }) if cooling(tripped_at_secs, now_secs) => {
            Decision::Cooling {
                until: cooldown_end(tripped_at_secs),
            }
        }
        _ => {
            let armed = Marker::Armed {
                pid: startup.pid,
                armed_at_secs: now_secs,
            };
            match write_marker(&path, armed) {
                Ok(()) => Decision::Armed,
                Err(_) => Decision::Unarmed,
            }
        }
    }
}

/// The production entry point: apply the breaker to the QUIC `settings`
/// chosen by the config, returning `None` when the breaker keeps QUIC off.
/// Logs the outcome and updates the transport telemetry. `now` is the wall
/// clock of the caller's `TimeSource`.
pub(crate) fn gate<T>(
    settings: Option<T>,
    data_dir: &Path,
    ignore_breaker: bool,
    now: SystemTime,
) -> Option<T> {
    if !ENABLED.load(Ordering::Relaxed) {
        return settings;
    }
    let stop_note = std::env::var_os(STOP_NOTE_ENV_VAR).map(PathBuf::from);
    let decision = decide(
        settings.is_some(),
        &Startup {
            data_dir,
            now,
            pid: std::process::id(),
            ignore_breaker,
            stop_note: stop_note.as_deref(),
        },
    );
    let metrics = &crate::transport::TRANSPORT_METRICS;
    let until_utc = |until: SystemTime| chrono::DateTime::<chrono::Utc>::from(until).to_rfc3339();
    match decision {
        Decision::NotRequested | Decision::Ignored => {}
        Decision::Armed => {
            // One build per process in production; a second keeps the first.
            let _first_wins = ARMED.set(marker_path(data_dir));
        }
        Decision::Unarmed => tracing::warn!(
            path = %marker_path(data_dir).display(),
            "Could not write the QUIC crash-breaker marker; QUIC runs without it"
        ),
        Decision::Tripped { until } => {
            metrics.record_quic_breaker_trip();
            metrics.set_quic_breaker_suppressed(true);
            tracing::warn!(
                until = %until_utc(until),
                cooldown_hours = QUIC_BREAKER_COOLDOWN.as_secs() / 3600,
                "The previous run with QUIC enabled ended abnormally, so QUIC stays off \
                 until the cooldown ends. Set `quic-ignore-crash-breaker = true` to run \
                 QUIC anyway"
            );
        }
        Decision::Cooling { until } => {
            metrics.set_quic_breaker_suppressed(true);
            tracing::warn!(
                until = %until_utc(until),
                "QUIC stays off: a run with QUIC enabled ended abnormally and the \
                 cooldown has not ended. Set `quic-ignore-crash-breaker = true` to run \
                 QUIC anyway"
            );
        }
    }
    if decision.quic_enabled() {
        settings
    } else {
        None
    }
}

fn unix_secs(time: SystemTime) -> u64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn cooldown_end(tripped_at_secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH
        + Duration::from_secs(tripped_at_secs.saturating_add(QUIC_BREAKER_COOLDOWN.as_secs()))
}

/// Whether a trip at `tripped_at_secs` still holds at `now_secs`. A trip time
/// more than a cooldown in the future (a clock set back a long way) is not
/// honoured, so it cannot hold QUIC off indefinitely.
fn cooling(tripped_at_secs: u64, now_secs: u64) -> bool {
    let cooldown = QUIC_BREAKER_COOLDOWN.as_secs();
    now_secs < tripped_at_secs.saturating_add(cooldown)
        && tripped_at_secs <= now_secs.saturating_add(cooldown)
}

/// Read and remove the run wrapper's stop note, returning the pid it names.
fn take_stop_note(note: &Path) -> Option<u32> {
    let content = std::fs::read_to_string(note).ok()?;
    let _consumed = std::fs::remove_file(note);
    content.trim().parse().ok()
}

fn write_marker(path: &Path, marker: Marker) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(&marker).map_err(std::io::Error::other)?;
    write_atomic(path, &bytes)
}

/// Write `bytes` to a sibling temporary file, then rename it into place.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("marker path has no parent"))?;
    std::fs::create_dir_all(dir)?;
    let name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("marker path has no file name"))?
        .to_string_lossy();
    let tmp = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PID: u32 = 4242;
    const T0: u64 = 1_800_000_000;

    fn at(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn start(dir: &Path, now: u64) -> Decision {
        decide(
            true,
            &Startup {
                data_dir: dir,
                now: at(now),
                pid: PID,
                ignore_breaker: false,
                stop_note: None,
            },
        )
    }

    fn read_marker(dir: &Path) -> Option<Marker> {
        let bytes = std::fs::read(marker_path(dir)).ok()?;
        Some(serde_json::from_slice(&bytes).expect("marker parses"))
    }

    /// What `quic_breaker_clean_exit` does for the marker this process armed.
    fn clean_exit(dir: &Path) {
        std::fs::remove_file(marker_path(dir)).expect("an armed marker to remove");
    }

    #[test]
    fn first_start_arms_the_marker() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(start(dir.path(), T0), Decision::Armed);
        assert_eq!(
            read_marker(dir.path()),
            Some(Marker::Armed {
                pid: PID,
                armed_at_secs: T0
            })
        );
    }

    #[test]
    fn an_unclean_marker_trips_the_breaker() {
        let dir = tempfile::tempdir().unwrap();
        start(dir.path(), T0);
        // The process ended without a clean exit: the marker is still armed.
        let decision = start(dir.path(), T0 + 30);
        assert_eq!(
            decision,
            Decision::Tripped {
                until: at(T0 + 30 + QUIC_BREAKER_COOLDOWN.as_secs())
            }
        );
        assert!(!decision.quic_enabled());
        assert_eq!(
            read_marker(dir.path()),
            Some(Marker::Tripped {
                tripped_at_secs: T0 + 30
            }),
            "the trip time is recorded"
        );
    }

    #[test]
    fn restarts_within_the_cooldown_stay_off_without_rearming() {
        let dir = tempfile::tempdir().unwrap();
        start(dir.path(), T0);
        start(dir.path(), T0 + 30);
        let tripped = read_marker(dir.path());
        let until = at(T0 + 30 + QUIC_BREAKER_COOLDOWN.as_secs());
        for later in [
            T0 + 31,
            T0 + 3600,
            T0 + 30 + QUIC_BREAKER_COOLDOWN.as_secs() - 1,
        ] {
            let decision = start(dir.path(), later);
            assert_eq!(decision, Decision::Cooling { until }, "at {later}");
            assert!(!decision.quic_enabled());
            assert_eq!(read_marker(dir.path()), tripped, "the marker is unchanged");
        }
    }

    #[test]
    fn quic_returns_after_the_cooldown() {
        let dir = tempfile::tempdir().unwrap();
        start(dir.path(), T0);
        start(dir.path(), T0 + 30);
        let after = T0 + 30 + QUIC_BREAKER_COOLDOWN.as_secs();
        assert_eq!(start(dir.path(), after), Decision::Armed);
        assert_eq!(
            read_marker(dir.path()),
            Some(Marker::Armed {
                pid: PID,
                armed_at_secs: after
            })
        );
    }

    #[test]
    fn a_clean_exit_does_not_trip() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(start(dir.path(), T0), Decision::Armed);
        clean_exit(dir.path());
        assert_eq!(start(dir.path(), T0 + 30), Decision::Armed);
    }

    #[test]
    fn quic_off_writes_no_marker() {
        let dir = tempfile::tempdir().unwrap();
        let decision = decide(
            false,
            &Startup {
                data_dir: dir.path(),
                now: at(T0),
                pid: PID,
                ignore_breaker: false,
                stop_note: None,
            },
        );
        assert_eq!(decision, Decision::NotRequested);
        assert!(!decision.quic_enabled());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn the_override_ignores_the_breaker() {
        let dir = tempfile::tempdir().unwrap();
        start(dir.path(), T0);
        start(dir.path(), T0 + 30);
        let tripped = read_marker(dir.path());
        let decision = decide(
            true,
            &Startup {
                data_dir: dir.path(),
                now: at(T0 + 60),
                pid: PID,
                ignore_breaker: true,
                stop_note: None,
            },
        );
        assert_eq!(decision, Decision::Ignored);
        assert!(decision.quic_enabled());
        assert_eq!(read_marker(dir.path()), tripped, "the marker is untouched");
    }

    /// The auto-update exit goes through the binary's clean-exit path, so it
    /// is `a_clean_exit_does_not_trip`. The run wrapper's update and restart
    /// stop the child with a hard kill instead, after noting its pid.
    #[test]
    fn a_wrapper_restart_or_update_is_not_a_trip() {
        let dir = tempfile::tempdir().unwrap();
        let note = dir.path().join("stop-note");
        start(dir.path(), T0);
        record_deliberate_stop(&note, PID);
        let decision = decide(
            true,
            &Startup {
                data_dir: dir.path(),
                now: at(T0 + 30),
                pid: PID + 1,
                ignore_breaker: false,
                stop_note: Some(&note),
            },
        );
        assert_eq!(decision, Decision::Armed);
        assert!(!note.exists(), "the note is consumed");

        // A note naming a different process does not excuse this one.
        record_deliberate_stop(&note, PID);
        let decision = decide(
            true,
            &Startup {
                data_dir: dir.path(),
                now: at(T0 + 60),
                pid: PID + 2,
                ignore_breaker: false,
                stop_note: Some(&note),
            },
        );
        assert!(
            matches!(decision, Decision::Tripped { .. }),
            "the marker was armed by pid {}, the note names {PID}: {decision:?}",
            PID + 1
        );
    }

    #[test]
    fn an_unreadable_marker_trips() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(marker_path(dir.path()), b"not json").unwrap();
        assert!(matches!(start(dir.path(), T0), Decision::Tripped { .. }));
    }

    #[test]
    fn a_trip_far_in_the_future_does_not_hold_quic_off() {
        let dir = tempfile::tempdir().unwrap();
        let future = T0 + 10 * QUIC_BREAKER_COOLDOWN.as_secs();
        write_marker(
            &marker_path(dir.path()),
            Marker::Tripped {
                tripped_at_secs: future,
            },
        )
        .unwrap();
        assert_eq!(start(dir.path(), T0), Decision::Armed);
    }

    /// `gate` passes settings through untouched while the breaker is not
    /// enabled, which is every test and embedder: no marker is written.
    #[test]
    fn gate_is_inert_unless_enabled() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!ENABLED.load(Ordering::Relaxed));
        assert_eq!(gate(Some(7u8), dir.path(), false, at(T0)), Some(7));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    /// The format is plain JSON with a `state` tag; pinned so a later release
    /// can read what this one wrote.
    #[test]
    fn marker_format() {
        let armed = serde_json::to_value(Marker::Armed {
            pid: 1,
            armed_at_secs: 2,
        })
        .unwrap();
        assert_eq!(
            armed,
            serde_json::json!({"state": "armed", "pid": 1, "armed_at_secs": 2})
        );
        let tripped = serde_json::to_value(Marker::Tripped { tripped_at_secs: 3 }).unwrap();
        assert_eq!(
            tripped,
            serde_json::json!({"state": "tripped", "tripped_at_secs": 3})
        );
    }
}
