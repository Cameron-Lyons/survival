use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");

    let extension_module = env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some();
    if extension_module {
        // An extension module resolves the Python C API from the interpreter that
        // loads it. maturin passes these linker flags itself; this keeps a direct
        // `cargo build --features extension-module` linkable on macOS.
        add_extension_module_link_args();
    }
    let plain_cargo_extension_module =
        extension_module && env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_none();
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

#[cfg(feature = "python")]
fn add_extension_module_link_args() {
    pyo3_build_config::add_extension_module_link_args();
}

#[cfg(not(feature = "python"))]
fn link_libpython() {}

#[cfg(not(feature = "python"))]
fn add_extension_module_link_args() {}
