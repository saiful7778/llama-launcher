use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::Emitter;
use tauri::Manager;

use crate::app_state::AppState;
use crate::data_types::{AppConfig, BinaryInfo, ModelInfo, ServerConfig};

#[tauri::command]
pub fn save_config(app: tauri::AppHandle, config: AppConfig) -> Result<(), String> {
    let config_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;

    fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;

    let config_path = config_dir.join("config.json");
    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;

    fs::write(config_path, json).map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub fn load_config(app: tauri::AppHandle) -> Result<AppConfig, String> {
    let config_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;

    let config_path = config_dir.join("config.json");

    if !config_path.exists() {
        return Ok(AppConfig::default());
    }

    let json = fs::read_to_string(config_path).map_err(|e| e.to_string())?;
    let config: AppConfig = serde_json::from_str(&json).map_err(|e| e.to_string())?;

    Ok(config)
}

#[tauri::command]
pub fn list_models(models_path: String) -> Result<Vec<ModelInfo>, String> {
    let base = PathBuf::from(&models_path);
    if !base.exists() {
        return Err(format!("Models directory does not exist: {}", models_path));
    }
    let mut models: Vec<ModelInfo> = Vec::new();

    find_gguf_files(&base, &base, &mut models)?;

    models
        .sort_by(|a: &ModelInfo, b: &ModelInfo| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    Ok(models)
}

fn find_gguf_files(
    base: &PathBuf,
    dir: &PathBuf,
    models: &mut Vec<ModelInfo>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            find_gguf_files(base, &path, models)?;
        } else if let Some(ext) = path.extension() {
            if ext == "gguf" {
                let relative = path.strip_prefix(base).unwrap_or(&path);

                models.push(ModelInfo {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }
    Ok(())
}

#[tauri::command]
pub fn check_binaries(llama_cpp_path: String) -> Result<Vec<BinaryInfo>, String> {
    let path = PathBuf::from(&llama_cpp_path);
    if !path.exists() {
        return Err(format!(
            "llama.cpp build directory does not exist: {}",
            llama_cpp_path
        ));
    }

    let mut binaries = Vec::new();
    for (name, desc) in known_binaries() {
        let binary_path = path.join(name);
        let exists = if cfg!(target_os = "windows") {
            binary_path.with_extension("exe").exists() || binary_path.exists()
        } else {
            binary_path.exists()
        };
        binaries.push(BinaryInfo {
            name: name.to_string(),
            path: binary_path.to_string_lossy().to_string(),
            exists,
            description: desc.to_string(),
        });
    }
    Ok(binaries)
}

fn known_binaries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "llama-server",
            "Start an OpenAI-compatible API server for model inference",
        ),
        (
            "llama-cli",
            "Run model inference in the terminal with interactive chat",
        ),
        (
            "llama-bench",
            "Benchmark model performance across different configurations",
        ),
        (
            "llama-quantize",
            "Quantize models to smaller formats (Q4, Q5, Q8, etc.)",
        ),
        (
            "llama-perplexity",
            "Compute perplexity of a model on given text data",
        ),
        ("llama-tokenize", "Tokenize text using a model's tokenizer"),
        ("llama-gguf-split", "Split or merge GGUF model files"),
    ]
}

#[tauri::command]
pub fn start_llama_server(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    config: ServerConfig,
) -> Result<(), String> {
    // Kill any existing server process first
    {
        let mut process = state.server_process.lock().map_err(|e| e.to_string())?;
        if let Some(ref mut child) = *process {
            let _ = child.kill();
            let _ = child.wait();
        }
        *process = None;
    }

    // Build argument list
    let mut args: Vec<String> = Vec::new();

    args.push("-m".to_string());
    args.push(config.model.clone());

    args.push("--host".to_string());
    args.push(config.host.clone());

    args.push("--port".to_string());
    args.push(config.port.to_string());

    args.push("-ngl".to_string());
    args.push(config.ngl.to_string());

    args.push("-c".to_string());
    args.push(config.ctx_size.to_string());

    args.push("-t".to_string());
    args.push(config.threads.to_string());

    if config.jinja {
        args.push("--jinja".to_string());
    }

    if let Some(ref draft_model) = config.spec_draft_model {
        args.push("--spec-draft-model".to_string());
        args.push(draft_model.clone());
        if let Some(n_max) = config.spec_draft_n_max {
            args.push("--spec-draft-n-max".to_string());
            args.push(n_max.to_string());
        }
    }

    // Resolve binary path
    let binary_name = if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let binary_path = PathBuf::from(&config.llama_cpp_path).join(binary_name);

    if !binary_path.exists() {
        return Err(format!(
            "llama-server binary not found at: {}",
            binary_path.display()
        ));
    }

    // Spawn the process
    let mut child = Command::new(&binary_path)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start llama-server: {}", e))?;

    let exit_emitted = Arc::new(AtomicBool::new(false));

    // Read stdout in background thread
    if let Some(stdout) = child.stdout.take() {
        let app_clone = app.clone();
        let exit_flag = exit_emitted.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let _ = app_clone.emit("server-log", &line);
                }
            }
            if !exit_flag.swap(true, Ordering::SeqCst) {
                let _ = app_clone.emit("server-exited", ());
            }
        });
    }

    // Read stderr in background thread
    if let Some(stderr) = child.stderr.take() {
        let app_clone = app.clone();
        let exit_flag = exit_emitted.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let _ = app_clone.emit("server-log", &line);
                }
            }
            if !exit_flag.swap(true, Ordering::SeqCst) {
                let _ = app_clone.emit("server-exited", ());
            }
        });
    }

    // Store child process handle
    {
        let mut process = state.server_process.lock().map_err(|e| e.to_string())?;
        *process = Some(child);
    }

    let _ = app.emit("server-started", ());
    Ok(())
}

#[tauri::command]
pub fn stop_llama_server(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut process = state.server_process.lock().map_err(|e| e.to_string())?;
    if let Some(ref mut child) = *process {
        child
            .kill()
            .map_err(|e| format!("Failed to kill server: {}", e))?;
        let _ = child.wait();
        *process = None;
    }
    Ok(())
}

#[tauri::command]
pub fn is_llama_server_running(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let mut process = state.server_process.lock().map_err(|e| e.to_string())?;
    match *process {
        Some(ref mut child) => match child.try_wait() {
            Ok(Some(_)) => {
                *process = None;
                Ok(false)
            }
            Ok(None) => Ok(true),
            Err(_) => {
                *process = None;
                Ok(false)
            }
        },
        None => Ok(false),
    }
}

#[tauri::command]
pub fn open_url(url: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&url)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(&["/C", "start", &url])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
