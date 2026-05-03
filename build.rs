use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");

    // `extension-module` suppresses normal libpython linkage. Rust test binaries
    // still need explicit libpython symbols on Unix platforms, but wheel builds
    // must not link libpython (manylinux policy).
    if env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_none() {
        return;
    }
    if env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some() {
        return;
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS");
    if !matches!(target_os.as_deref(), Ok("linux" | "macos")) {
        return;
    }

    #[cfg(not(feature = "python"))]
    {}

    #[cfg(feature = "python")]
    {
        let config = pyo3_build_config::get();

        if let Some(lib_dir) = &config.lib_dir {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        if let Some(lib_name) = &config.lib_name {
            println!("cargo:rustc-link-lib={lib_name}");
        }
    }
}
