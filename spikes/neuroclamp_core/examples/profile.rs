//! Run one preset a few times, for use under a profiler.
//!
//! ```bash
//! CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --no-default-features --example profile
//! valgrind --tool=callgrind target/release/examples/profile purkinje
//! callgrind_annotate --inclusive=yes callgrind.out.<pid> | head -40
//! ```

use std::fs;
use std::hint::black_box;
use std::path::Path;

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

fn main() {
    let filter = std::env::args()
        .nth(1)
        .expect("usage: profile <preset-name-substring>");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_fixtures");
    let fixture = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter_map(|p| serde_json::from_str::<Fixture>(&fs::read_to_string(p).ok()?).ok())
        .find(|f| f.name.to_lowercase().contains(&filter.to_lowercase()))
        .expect("no fixture matches");
    let model = Model::from_json(&fixture.spec_json).unwrap();

    let mut i_ext = vec![0.0; 400];
    i_ext.extend(std::iter::repeat_n(8.0, 4000));
    i_ext.extend(std::iter::repeat_n(0.0, 1600));
    for _ in 0..3 {
        black_box(
            model
                .simulate_current_clamp(
                    &i_ext,
                    fixture.v0,
                    &fixture.gates0,
                    fixture.ca0,
                    fixture.dt,
                )
                .unwrap(),
        );
    }
    eprintln!("profiled {}", fixture.name);
}
