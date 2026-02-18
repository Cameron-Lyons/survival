use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");

    // `extension-module` suppresses normal libpython linkage. Rust test binaries
    // still need explicit libpython symbols on Linux.
    if env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_none() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let config = pyo3_build_config::get();

    if let Some(lib_dir) = &config.lib_dir {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }
    if let Some(lib_name) = &config.lib_name {
        println!("cargo:rustc-link-lib={lib_name}");
    }
}
