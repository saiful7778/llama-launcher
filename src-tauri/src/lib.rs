pub mod app_state;
pub mod commands;
pub mod data_types;

use crate::app_state::AppState;
use crate::commands::{
    check_binaries, is_llama_server_running, list_models, load_config, open_url, save_config,
    start_llama_server, stop_llama_server,
};
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::new()) // Use ::new() instead of direct struct construction
        .invoke_handler(tauri::generate_handler![
            save_config,
            load_config,
            list_models,
            check_binaries,
            start_llama_server,
            stop_llama_server,
            is_llama_server_running,
            open_url,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<AppState>();
                let mut process = state.server_process.lock().unwrap();
                if let Some(ref mut child) = *process {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                *process = None;
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
