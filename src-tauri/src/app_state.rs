use std::process::Child;
use std::sync::Mutex;

pub struct AppState {
    pub server_process: Mutex<Option<Child>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            server_process: Mutex::new(None),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
