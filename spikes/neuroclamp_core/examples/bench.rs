//! Benchmark the current-clamp integrator without Python.
//!
//! Loads every fixture in `bench_fixtures/` (written by
//! `spikes/rate_expr/dump_bench_fixtures.py`) and times a 150 ms current step
//! (10 ms baseline, 100 ms at 8 µA/cm², 40 ms recovery): the same protocol
//! `spikes/rate_expr/verify_rust.py` uses.
//!
//! ```bash
//! cargo run --release --no-default-features --example bench            # all presets
//! cargo run --release --no-default-features --example bench -- purkinje # name filter
//! ```

use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use neuroclamp_core::current_clamp::Model;
use serde::Deserialize;

/// One preset's spec and initial state, as written by the Python dump script.
#[derive(Deserialize)]
struct Fixture {
    name: String,
    spec_json: String,
    v0: f64,
    gates0: Vec<f64>,
    ca0: f64,
    dt: f64,
}

/// The 150 ms step protocol: 400 + 4000 + 1600 samples at 40 kHz.
fn step_protocol() -> Vec<f64> {
    let mut i_ext = vec![0.0; 400];
    i_ext.extend(std::iter::repeat_n(8.0, 4000));
    i_ext.extend(std::iter::repeat_n(0.0, 1600));
    i_ext
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default().to_lowercase();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_fixtures");
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();

    let i_ext = step_protocol();
    println!("{:36} {:>10} {:>12}", "preset", "ns/step", "ms/150 ms");
    for path in paths {
        let fixture: Fixture = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        if !fixture.name.to_lowercase().contains(&filter) {
            continue;
        }
        let model = Model::from_json(&fixture.spec_json).unwrap();
        let run = || {
            model
                .simulate_current_clamp(
                    black_box(&i_ext),
                    fixture.v0,
                    &fixture.gates0,
                    fixture.ca0,
                    fixture.dt,
                )
                .unwrap()
        };
        black_box(run()); // warm up caches and branch predictors

        // Best of several batches, each long enough (~0.2 s) to swamp timer noise.
        let reps = ((0.2 / time_one(&run)).ceil() as usize).max(1);
        let best = (0..5)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..reps {
                    black_box(run());
                }
                start.elapsed().as_secs_f64() / reps as f64
            })
            .fold(f64::INFINITY, f64::min);
        println!(
            "{:36} {:>10.0} {:>12.2}",
            fixture.name,
            best / i_ext.len() as f64 * 1e9,
            best * 1e3
        );
    }
}

/// Wall time of a single call, in seconds.
fn time_one<T>(run: &impl Fn() -> T) -> f64 {
    let start = Instant::now();
    black_box(run());
    start.elapsed().as_secs_f64()
}
