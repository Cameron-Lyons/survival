//! Links `libpython` into the crate only when a plain `cargo` invocation
//! enables the `extension-module` feature.
//!
//! PyO3 env contract (see `pyo3-build-config`):
//! - `PYO3_PYTHON` selects the interpreter whose configuration PyO3 reads; the
//!   `python` feature alone lets PyO3 link `libpython` itself, which is what
//!   `cargo test --features python[,ml]` relies on.
//! - `extension-module` (`pyo3/extension-module`) tells PyO3 *not* to link
//!   `libpython`, because a wheel's `.so` must resolve the interpreter's
//!   symbols at import time. That leaves a `cargo test --features
//!   extension-module` (or `--all-features`) binary without an interpreter, so
//!   this script adds the link back in exactly that case.
//! - `PYO3_BUILD_EXTENSION_MODULE=1` is set by maturin for every wheel and
//!   `maturin develop` build; when it is present nothing is linked, so the
//!   shipped `.so` never depends on `libpython` (`ldd` shows no libpython
//!   entry). Set it yourself when building a cdylib with bare `cargo build
//!   --features extension-module`.
//! - `PYO3_CONFIG_FILE` overrides the interpreter configuration entirely;
//!   the rerun directives keep the link decision in step with it.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");

    let plain_cargo_extension_module = env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some()
        && env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_none();
    // Windows always links through PyO3's raw-dylib import entries; other
    // targets are not test hosts for this crate.
    let unix_test_host = matches!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("linux" | "macos")
    );
    if plain_cargo_extension_module && unix_test_host {
        link_libpython();
    }
}

#[cfg(feature = "python")]
fn link_libpython() {
    let config = pyo3_build_config::get();
    if let Some(lib_dir) = config.lib_dir() {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }
    if let Some(lib_name) = config.lib_name() {
        println!("cargo:rustc-link-lib={lib_name}");
    }
}

// `extension-module` implies `python`, so this arm is unreachable in practice;
// it keeps the script compiling when the feature set is inconsistent.
#[cfg(not(feature = "python"))]
fn link_libpython() {}
